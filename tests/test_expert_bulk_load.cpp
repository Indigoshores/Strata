// tests/test_expert_bulk_load.cpp - Pass 5.1 Bit-Exact Bulk Upload & Prefetch Test
#include "strata/core/expert_cache.hpp"
#include "strata/platform/memory.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

constexpr size_t kNumExperts = 64;
constexpr size_t kExpertBytes = 2473984; // IQ4_XS typical expert size (18191 * 136)
constexpr size_t kTotalBytes = kNumExperts * kExpertBytes;

struct SplitMix64 {
    uint64_t s = 0x853c49e6748fea9bULL;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

} // namespace

int main(int argc, char** argv) {
    (void) argc; (void) argv;
    std::puts("=== Strata Pass 5.1: test_expert_bulk_load ===");

    // Step 1: Create a synthetic expert file with deterministic splitmix64 contents
    std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "strata_test_bulk_experts.tmp";
    std::printf("[1/6] Generating %zu synthetic experts (%zu bytes each, total %.2f MiB) at %s\n",
                kNumExperts, kExpertBytes, (double) kTotalBytes / (1024.0 * 1024.0), temp_path.string().c_str());

    {
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "Failed to create temporary file: %s\n", temp_path.string().c_str());
            return 1;
        }

        SplitMix64 rng;
        constexpr size_t kChunkWords = 32768; // 256 KiB chunk
        std::vector<uint64_t> chunk(kChunkWords);
        size_t words_written = 0;
        const size_t total_words = kTotalBytes / sizeof(uint64_t);

        while (words_written < total_words) {
            size_t n = std::min(kChunkWords, total_words - words_written);
            for (size_t i = 0; i < n; ++i) {
                chunk[i] = rng.next();
            }
            out.write(reinterpret_cast<const char*>(chunk.data()), n * sizeof(uint64_t));
            words_written += n;
        }
        // Write any remainder bytes
        size_t rem = kTotalBytes % sizeof(uint64_t);
        if (rem > 0) {
            uint64_t val = rng.next();
            out.write(reinterpret_cast<const char*>(&val), rem);
        }
        out.close();
    }

    // Step 2: Memory map the temporary file
    std::puts("[2/6] Memory-mapping synthetic expert file...");
    const uint8_t* mapped_data = nullptr;
#if defined(_WIN32)
    HANDLE hFile = CreateFileA(temp_path.string().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "CreateFileA failed, error %lu\n", GetLastError());
        std::filesystem::remove(temp_path);
        return 1;
    }
    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (hMap == nullptr) {
        std::fprintf(stderr, "CreateFileMappingA failed, error %lu\n", GetLastError());
        CloseHandle(hFile);
        std::filesystem::remove(temp_path);
        return 1;
    }
    mapped_data = reinterpret_cast<const uint8_t*>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
    if (mapped_data == nullptr) {
        std::fprintf(stderr, "MapViewOfFile failed, error %lu\n", GetLastError());
        CloseHandle(hMap);
        CloseHandle(hFile);
        std::filesystem::remove(temp_path);
        return 1;
    }
#else
    int fd = open(temp_path.string().c_str(), O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "open failed\n");
        std::filesystem::remove(temp_path);
        return 1;
    }
    mapped_data = reinterpret_cast<const uint8_t*>(mmap(nullptr, kTotalBytes, PROT_READ, MAP_SHARED, fd, 0));
    if (mapped_data == MAP_FAILED) {
        std::fprintf(stderr, "mmap failed\n");
        close(fd);
        std::filesystem::remove(temp_path);
        return 1;
    }
#endif

    // Step 3: Test prefetch_ranges
    std::puts("[3/6] Testing batched prefetch_ranges...");
    std::vector<std::pair<const void*, size_t>> ranges;
    ranges.reserve(kNumExperts);
    for (size_t i = 0; i < kNumExperts; ++i) {
        ranges.emplace_back(mapped_data + i * kExpertBytes, kExpertBytes);
    }
    strata::platform::prefetch_ranges(ranges.data(), ranges.size());
    std::puts("      prefetch_ranges completed successfully.");

    // Step 4: Bulk upload via BulkUploadSession into cache_bulk
    std::puts("[4/6] Uploading experts via ExpertCache::BulkUploadSession...");
    strata::core::ExpertCache cache_bulk;
    std::string err;
    if (!cache_bulk.open(kNumExperts, 1, kNumExperts, kExpertBytes, err)) {
        std::fprintf(stderr, "cache_bulk.open failed: %s\n", err.c_str());
        goto cleanup_mmap;
    }

    {
        strata::core::ExpertCache::BulkUploadSession session(cache_bulk);
        if (!session.init(err)) {
            std::fprintf(stderr, "session.init failed: %s\n", err.c_str());
            goto cleanup_mmap;
        }

        std::vector<strata::core::ExpertCache::BulkUploadSession::Item> items;
        items.reserve(kNumExperts);
        for (size_t i = 0; i < kNumExperts; ++i) {
            int32_t slot = cache_bulk.admit(0, static_cast<int64_t>(i));
            if (slot < 0) {
                std::fprintf(stderr, "cache_bulk.admit failed for expert %zu\n", i);
                goto cleanup_mmap;
            }
            items.push_back({slot, mapped_data + i * kExpertBytes, static_cast<int64_t>(kExpertBytes)});
        }

        if (!session.upload(items.data(), items.size(), err)) {
            std::fprintf(stderr, "session.upload failed: %s\n", err.c_str());
            goto cleanup_mmap;
        }

        if (!session.finish(err)) {
            std::fprintf(stderr, "session.finish failed: %s\n", err.c_str());
            goto cleanup_mmap;
        }

        // Verify post-upload 0 B pinned buffer retention
        if (session.retained_staging_bytes() != 0) {
            std::fprintf(stderr, "ERROR: BulkUploadSession retained %zu bytes of pinned staging memory (expected 0 B)!\n",
                         session.retained_staging_bytes());
            goto cleanup_mmap;
        }
        std::puts("      BulkUploadSession finished: 0 B pinned buffer retained.");
    }

    // Step 5: Upload reference via legacy fill_slot_blocking into cache_ref
    std::puts("[5/6] Uploading reference experts via legacy fill_slot_blocking...");
    {
        strata::core::ExpertCache cache_ref;
        if (!cache_ref.open(kNumExperts, 1, kNumExperts, kExpertBytes, err)) {
            std::fprintf(stderr, "cache_ref.open failed: %s\n", err.c_str());
            goto cleanup_mmap;
        }

        for (size_t i = 0; i < kNumExperts; ++i) {
            int32_t slot = cache_ref.admit(0, static_cast<int64_t>(i));
            if (slot < 0 || !cache_ref.fill_slot_blocking(slot, mapped_data + i * kExpertBytes, err, kExpertBytes)) {
                std::fprintf(stderr, "cache_ref.fill_slot_blocking failed for expert %zu: %s\n", i, err.c_str());
                goto cleanup_mmap;
            }
        }

        // Step 6: Verify 100% byte-for-byte VRAM parity between bulk and reference, and against source
        std::puts("[6/6] Verifying 100% byte-for-byte VRAM parity across all 64 slots...");
        std::vector<uint8_t> bulk_host(kExpertBytes);
        std::vector<uint8_t> ref_host(kExpertBytes);

        for (size_t i = 0; i < kNumExperts; ++i) {
            const uint8_t* bulk_dev = cache_bulk.device_slot(static_cast<int32_t>(i));
            const uint8_t* ref_dev = cache_ref.device_slot(static_cast<int32_t>(i));

            if (!bulk_dev || !ref_dev) {
                std::fprintf(stderr, "ERROR: Null device slot at index %zu\n", i);
                goto cleanup_mmap;
            }

            cudaError_t e1 = cudaMemcpy(bulk_host.data(), bulk_dev, kExpertBytes, cudaMemcpyDeviceToHost);
            cudaError_t e2 = cudaMemcpy(ref_host.data(), ref_dev, kExpertBytes, cudaMemcpyDeviceToHost);
            if (e1 != cudaSuccess || e2 != cudaSuccess) {
                std::fprintf(stderr, "ERROR: cudaMemcpy D2H failed at index %zu\n", i);
                goto cleanup_mmap;
            }

            // Compare bulk against reference
            if (std::memcmp(bulk_host.data(), ref_host.data(), kExpertBytes) != 0) {
                std::fprintf(stderr, "ERROR: Bulk upload does not match reference at slot %zu!\n", i);
                std::fprintf(stderr, "bulk_host[0..7]: ");
                for (int b = 0; b < 8; ++b) std::fprintf(stderr, "%02x ", bulk_host[b]);
                std::fprintf(stderr, "\nref_host[0..7]:  ");
                for (int b = 0; b < 8; ++b) std::fprintf(stderr, "%02x ", ref_host[b]);
                std::fprintf(stderr, "\nmapped[0..7]:    ");
                for (int b = 0; b < 8; ++b) std::fprintf(stderr, "%02x ", mapped_data[i * kExpertBytes + b]);
                std::fprintf(stderr, "\n");
                goto cleanup_mmap;
            }

            // Compare bulk against mapped source
            if (std::memcmp(bulk_host.data(), mapped_data + i * kExpertBytes, kExpertBytes) != 0) {
                std::fprintf(stderr, "ERROR: Bulk upload does not match source file at slot %zu!\n", i);
                goto cleanup_mmap;
            }

            // Verify with cache_bulk.verify_slot
            if (!cache_bulk.verify_slot(static_cast<int32_t>(i), mapped_data + i * kExpertBytes, err, kExpertBytes)) {
                std::fprintf(stderr, "ERROR: verify_slot failed at slot %zu: %s\n", i, err.c_str());
                goto cleanup_mmap;
            }
        }
        std::puts("      100% byte-for-byte parity confirmed across all 64 slots (memcmp == 0)!");
        cache_ref.close();
    }

    cache_bulk.close();

#if defined(_WIN32)
    UnmapViewOfFile(mapped_data);
    CloseHandle(hMap);
    CloseHandle(hFile);
#else
    munmap(const_cast<uint8_t*>(mapped_data), kTotalBytes);
    close(fd);
#endif
    std::filesystem::remove(temp_path);
    std::puts("=== All Pass 5.1 Bulk Load & Prefetch Tests PASSED! ===");
    return 0;

cleanup_mmap:
    cache_bulk.close();
#if defined(_WIN32)
    if (mapped_data) UnmapViewOfFile(mapped_data);
    if (hMap) CloseHandle(hMap);
    if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
    if (mapped_data && mapped_data != MAP_FAILED) munmap(const_cast<uint8_t*>(mapped_data), kTotalBytes);
    if (fd >= 0) close(fd);
#endif
    std::filesystem::remove(temp_path);
    return 1;
}
