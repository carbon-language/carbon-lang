package main

import (
	"bufio"
	"debug/elf"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func symsizes(path string) map[string]float64 {
	m := make(map[string]float64)
	f, err := elf.Open(path)
	if err != nil {
		panic(err.Error())
	}
	syms, err := f.Symbols()
	if err != nil {
		panic(err.Error())
	}
	for _, sym := range syms {
		if sym.Section < elf.SectionIndex(len(f.Sections)) && f.Sections[sym.Section].Name == ".text" {
			m[sym.Name] = float64(sym.Size)
		}
	}
	return m
}

func benchnums(path, stat string) map[string]float64 {
	m := make(map[string]float64)

	fh, err := os.Open(path)
	if err != nil {
		panic(err.Error())
	}

	scanner := bufio.NewScanner(fh)
	for scanner.Scan() {
		elems := strings.Split(scanner.Text(), "\t")
		if !strings.HasPrefix(elems[0], "Benchmark") || len(elems) < 3 {
			continue
		}
		var s string
		for _, elem := range elems[2:] {
			selems := strings.Split(strings.TrimSpace(elem), " ")
			if selems[1] == stat {
				s = selems[0]
			}
		}
		if s != "" {
			ns, err := strconv.ParseFloat(s, 64)
			if err != nil {
				panic(scanner.Text() + " ---- " + err.Error())
			}
			m[elems[0]] = ns
		}
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}

	return m
}

func main() {
	var cmp func(string) map[string]float64
	switch os.Args[1] {
	case "symsizes":
		cmp = symsizes

	case "benchns":
		cmp = func(path string) map[string]float64 {
			return benchnums(path, "ns/op")
		}

	case "benchallocs":
		cmp = func(path string) map[string]float64 {
			return benchnums(path, "allocs/op")
		}
	}

	syms1 := cmp(os.Args[2])
	syms2 := cmp(os.Args[3])

	for n, z1 := range syms1 {
		if z2, ok := syms2[n]; ok && z2 != 0 {
			fmt.Printf("%s %f %f %f\n", n, z1, z2, z1/z2)
		}
	}
}
