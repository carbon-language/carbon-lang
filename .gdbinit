# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

source external/+llvm_project+llvm-project/llvm/utils/gdb-scripts/prettyprinters.py
source external/+llvm_project+llvm-project/libcxx/utils/gdb/libcxx/printers.py
python register_libcxx_printer_loader()
