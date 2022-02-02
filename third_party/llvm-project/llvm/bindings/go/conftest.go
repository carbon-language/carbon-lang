package main

import (
	"go/build"
	"os"
)

// Tests that the Go compiler is at least version 1.2.
func main() {
	for _, tag := range build.Default.ReleaseTags {
		if tag == "go1.2" {
			os.Exit(0)
		}
	}
	os.Exit(1)
}
