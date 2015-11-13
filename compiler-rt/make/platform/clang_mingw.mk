Description := Static runtime libraries for mingw-w64

###

CC ?= cc
AR ?= ar

Arch := unknown
Configs :=

SupportedArches := x86_64 i386 arm

Configs += builtins-x86_64 builtins-i386 builtins-arm
Arch.builtins-x86_64 := x86_64
Arch.builtins-i386 := i386
Arch.builtins-arm := arm

###

CFLAGS := -Wall -O3 -fomit-frame-pointer
CFLAGS.builtins-x86_64 := -target x86_64-windows-gnu $(CFLAGS)
CFLAGS.builtins-i386 := -target i686-windows-gnu $(CFLAGS)
CFLAGS.builtins-arm := -target armv7-windows-gnu $(CFLAGS)

FUNCTIONS.builtins-x86_64 := $(CommonFunctions) $(ArchFunctions.x86_64)
FUNCTIONS.builtins-i386 := $(CommonFunctions) $(ArchFunctions.i386)
FUNCTIONS.builtins-arm := $(CommonFunctions) $(ArchFunctions.arm)

# Always use optimized variants.
OPTIMIZED := 1
