# REQUIRES: system-linux

# RUN: mkdir -p %t.dir && cd %t.dir
# RUN: cp %S/Inputs/debug-fission-simple.s debug-fission-simple.s
# RUN: cp %S/Inputs/debug-fission-script.txt debug-fission-script.txt
# RUN: llvm-mc -g -filetype=obj -triple x86_64-unknown-unknown --split-dwarf-file=debug-fission-simple.dwo \
# RUN:   ./debug-fission-simple.s -o ./debug-fission-simple.o
# RUN: %host_cxx %cxxflags -g -Wl,--gc-sections,-q,-nostdlib -Wl,--undefined=_Z6_startv -nostartfiles -Wl,--script=debug-fission-script.txt ./debug-fission-simple.o -o out.exe
# RUN: llvm-bolt out.exe --reorder-blocks=reverse -update-debug-sections -dwo-output-path=%t.dir -o out.bolt
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.dir/debug-fission-simple.dwo0.dwo | grep DW_FORM_GNU_addr_index | FileCheck %s --check-prefix=CHECK-ADDR-INDEX
# RUN: llvm-dwarfdump --show-form --verbose --debug-addr out.bolt | FileCheck %s --check-prefix=CHECK-ADDR-SEC

# CHECK-ADDR-INDEX: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000001)
# CHECK-ADDR-INDEX: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000002)
# CHECK-ADDR-INDEX: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000003)

# CHECK-ADDR-SEC: .debug_addr contents:
# CHECK-ADDR-SEC: 0x00000000: Addrs: [
# CHECK-ADDR-SEC: 0x0000000000601000
# CHECK-ADDR-SEC: 0x0000000000a00000
# CHECK-ADDR-SEC: 0x0000000000000000
# CHECK-ADDR-SEC: 0x0000000000a00040

//clang++ -ffunction-sections -fno-exceptions -g -gsplit-dwarf=split -S debug-fission-simple.cpp -o debug-fission-simple.s
static int foo = 2;
int doStuff(int val) {
  if (val == 5)
    val += 1 + foo;
  else
    val -= 1;
  return val;
}

int doStuff2(int val) {
  return val += 3;
}

int main(int argc, const char** argv) {
  return doStuff(argc);
}
