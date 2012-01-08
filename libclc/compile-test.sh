#!/bin/sh

clang -ccc-host-triple ptx32--nvidiacl -Iptx-nvidiacl/include -Igeneric/include -Xclang -mlink-bitcode-file -Xclang ptx32--nvidiacl/lib/builtins.bc -include clc/clc.h -Dcl_clang_storage_class_specifiers "$@"
