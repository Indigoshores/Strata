package main

import (
	"bytes"
	"debug/pe"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

type ExportDirectory struct {
	Characteristics       uint32
	TimeDateStamp         uint32
	MajorVersion          uint16
	MinorVersion          uint16
	NameRVA               uint32
	OrdinalBase           uint32
	NumberOfFunctions     uint32
	NumberOfNames         uint32
	AddressOfFunctions    uint32
	AddressOfNames        uint32
	AddressOfNameOrdinals uint32
}

func rvaToOffset(f *pe.File, rva uint32) (uint32, error) {
	for _, sec := range f.Sections {
		size := sec.VirtualSize
		if size == 0 {
			size = sec.Size
		}
		if rva >= sec.VirtualAddress && rva < sec.VirtualAddress+size {
			return sec.Offset + (rva - sec.VirtualAddress), nil
		}
	}
	return 0, fmt.Errorf("RVA 0x%x not found in any section", rva)
}

func readCString(r io.ReaderAt, offset uint32) (string, error) {
	buf := make([]byte, 256)
	var str []byte
	cur := int64(offset)
	for {
		n, err := r.ReadAt(buf, cur)
		if n <= 0 && err != nil {
			return "", err
		}
		nullIdx := bytes.IndexByte(buf[:n], 0)
		if nullIdx != -1 {
			str = append(str, buf[:nullIdx]...)
			return string(str), nil
		}
		str = append(str, buf[:n]...)
		cur += int64(n)
	}
}

func getExportNames(filePath string) (string, []string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", nil, err
	}
	defer file.Close()

	peFile, err := pe.NewFile(file)
	if err != nil {
		return "", nil, err
	}
	defer peFile.Close()

	var exportDirRVA, exportDirSize uint32
	switch opt := peFile.OptionalHeader.(type) {
	case *pe.OptionalHeader64:
		if len(opt.DataDirectory) > 0 {
			exportDirRVA = opt.DataDirectory[0].VirtualAddress
			exportDirSize = opt.DataDirectory[0].Size
		}
	case *pe.OptionalHeader32:
		if len(opt.DataDirectory) > 0 {
			exportDirRVA = opt.DataDirectory[0].VirtualAddress
			exportDirSize = opt.DataDirectory[0].Size
		}
	default:
		return "", nil, fmt.Errorf("unknown optional header type")
	}

	if exportDirRVA == 0 || exportDirSize == 0 {
		return "", nil, fmt.Errorf("no export table found")
	}

	expOff, err := rvaToOffset(peFile, exportDirRVA)
	if err != nil {
		return "", nil, err
	}

	var ed ExportDirectory
	sr := io.NewSectionReader(file, int64(expOff), int64(binary.Size(ed)))
	if err := binary.Read(sr, binary.LittleEndian, &ed); err != nil {
		return "", nil, fmt.Errorf("reading export directory: %w", err)
	}

	var dllName string
	if ed.NameRVA != 0 {
		nameOff, err := rvaToOffset(peFile, ed.NameRVA)
		if err == nil {
			dllName, _ = readCString(file, nameOff)
		}
	}
	if dllName == "" {
		dllName = filepath.Base(filePath)
	}

	if ed.NumberOfNames == 0 {
		return dllName, nil, nil
	}

	namesOff, err := rvaToOffset(peFile, ed.AddressOfNames)
	if err != nil {
		return "", nil, fmt.Errorf("AddressOfNames offset: %w", err)
	}

	nameRVAs := make([]uint32, ed.NumberOfNames)
	srNames := io.NewSectionReader(file, int64(namesOff), int64(ed.NumberOfNames*4))
	if err := binary.Read(srNames, binary.LittleEndian, nameRVAs); err != nil {
		return "", nil, fmt.Errorf("reading name RVAs: %w", err)
	}

	var names []string
	for _, nRva := range nameRVAs {
		nOff, err := rvaToOffset(peFile, nRva)
		if err != nil {
			continue
		}
		symName, err := readCString(file, nOff)
		if err == nil && symName != "" {
			names = append(names, symName)
		}
	}

	sort.Strings(names)
	return dllName, names, nil
}

func main() {
	targetDLLs := []string{
		`C:\Users\Butch\Documents\10-Models\Strata\engine\rocm\bin\libhipblaslt.dll`,
		`C:\Users\Butch\Documents\10-Models\Strata\engine\rocm\bin\rocblas.dll`,
		`C:\Users\Butch\Documents\10-Models\Strata\engine\rocm\bin\hipblas.dll`,
		`C:\Users\Butch\Documents\10-Models\Strata\engine\amdhip64_7.dll`,
	}

	outDir := `C:\Users\Butch\Documents\02_Apps\Strata\rocm100500-libs`
	if err := os.MkdirAll(outDir, 0755); err != nil {
		fmt.Printf("Error creating %s: %v\n", outDir, err)
		os.Exit(1)
	}

	for _, dllPath := range targetDLLs {
		dllName, names, err := getExportNames(dllPath)
		if err != nil {
			fmt.Printf("Error extracting exports from %s: %v\n", dllPath, err)
			continue
		}
		fmt.Printf("Found %d exported symbols in %s (internal name: %s)\n", len(names), filepath.Base(dllPath), dllName)

		baseNoExt := filepath.Base(dllPath)
		if ext := filepath.Ext(baseNoExt); ext != "" {
			baseNoExt = baseNoExt[:len(baseNoExt)-len(ext)]
		}
		defFileName := filepath.Join(outDir, baseNoExt+".def")
		defContent := fmt.Sprintf("LIBRARY %s\nEXPORTS\n", dllName)
		for _, name := range names {
			defContent += fmt.Sprintf("    %s\n", name)
		}
		if err := os.WriteFile(defFileName, []byte(defContent), 0644); err != nil {
			fmt.Printf("Error writing %s: %v\n", defFileName, err)
		} else {
			fmt.Printf("  -> Wrote %s\n", defFileName)
		}
	}
}
