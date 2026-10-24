#pragma once
// Included only by STRATA_ENABLE_HIP builds. CUDA builds use NVIDIA headers.
#if defined(_WIN32)
// hip_runtime.h pulls in <windows.h> on Windows: keep its min/max macros and the rarely used APIs out of the engine.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
// Do not let HIP's legacy macro corrupt libstdc++ attribute names.
#ifdef __noinline__
#undef __noinline__
#endif
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrIntegrated hipDeviceAttributeIntegrated
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#if defined(_WIN32)
inline hipError_t strata_hip_event_synchronize(hipEvent_t ev) {
    (void)hipEventQuery(ev);
    return hipEventSynchronize(ev);
}
#define cudaEventSynchronize strata_hip_event_synchronize
#else
#define cudaEventSynchronize hipEventSynchronize
#endif
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#if HIP_VERSION_MAJOR < 7
#define cudaFreeHost hipHostFree
#else
#define cudaFreeHost hipFreeHost
#endif
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraph_t hipGraph_t
#define cudaHostAlloc hipHostMalloc
#if HIP_VERSION_MAJOR < 7
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostAllocPortable hipHostMallocPortable
#else
#define cudaHostAllocDefault hipHostAllocDefault
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostAllocPortable hipHostAllocPortable
#endif
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost(...) (::strata::hip_compat::malloc_host(__VA_ARGS__))
#if defined(_WIN32)
// Windows: hipMemGetInfo counts this process's own allocations only, so the desktop's and other programs' share of
// the card is invisible to it, and VRAM sized by it overfills the card (WDDM then moves memory out to system RAM and
// the GPU slows down).  Its free figure is lowered by what Windows' video memory budget for this process withholds
// (src/core/device.cu; STRATA_WDDM_BUDGET=0: hipMemGetInfo as it is).
namespace strata::hip_compat {
hipError_t mem_get_info(size_t* free_bytes, size_t* total_bytes);
}
#define cudaMemGetInfo ::strata::hip_compat::mem_get_info
#else
#define cudaMemGetInfo hipMemGetInfo
#endif
#define cudaMemcpy hipMemcpy
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#if defined(_WIN32)
extern "C" hipError_t strata_wddm_d2d_copy(void* dst, const void* src, size_t bytes, hipStream_t stream);

inline hipError_t strata_hip_memcpy_async(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream = nullptr) {
    if (kind == hipMemcpyDeviceToDevice && bytes > 0 && bytes <= (16ull << 20)) {
        static const bool s_use_kernel = []() {
            const char* env = std::getenv("STRATA_WDDM_COPY_KERNEL");
            return env ? (std::atoi(env) != 0) : true;
        }();
        if (s_use_kernel) {
            return strata_wddm_d2d_copy(dst, src, bytes, stream);
        }
    }
    return hipMemcpyAsync(dst, src, bytes, kind, stream);
}
#define cudaMemcpyAsync strata_hip_memcpy_async
#else
#define cudaMemcpyAsync hipMemcpyAsync
#endif
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpyPeer hipMemcpyPeer
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaSetDevice hipSetDevice
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
inline void strata_wddm_flush(hipStream_t stream) {
#if defined(_WIN32)
    static const bool s_flush = []() {
        const char* env = std::getenv("STRATA_WDDM_FLUSH");
        return env ? (std::atoi(env) != 0) : true;
    }();
    if (s_flush && stream != nullptr) {
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(stream, &capture) == hipSuccess && capture == hipStreamCaptureStatusNone) {
            (void)hipStreamQuery(stream);
        }
    }
#else
    (void)stream;
#endif
}
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

namespace strata::hip_compat {
template <typename T>
inline hipError_t malloc_host(T** pointer, size_t bytes) {
    return hipHostMalloc(reinterpret_cast<void**>(pointer), bytes, hipHostMallocDefault);
}
}  // namespace strata::hip_compat

template <typename Kernel>
inline hipError_t cudaFuncSetAttribute(Kernel kernel, hipFuncAttribute attribute, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph, unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                     hipGraphNode_t* error, char* log, size_t size) {
    return hipGraphInstantiate(exec, graph, error, log, size);
}
#define __trap() __builtin_trap()   // the compiled-out sm_80 paths (never selected on AMD)
#define cudaMemcpyToSymbol(symbol, ...) hipMemcpyToSymbol(HIP_SYMBOL(symbol), __VA_ARGS__)
#define cudaMemcpyFromSymbol(dst, symbol, ...) hipMemcpyFromSymbol(dst, HIP_SYMBOL(symbol), __VA_ARGS__)

#include "intrinsics.hpp"
