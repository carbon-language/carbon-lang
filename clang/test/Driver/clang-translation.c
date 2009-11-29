// RUN: clang -ccc-host-triple i386-unknown-unknown -### -S -O0 -Os %s -o %t.s -fverbose-asm -funwind-tables 2> %t.log
// RUN: grep '"-triple" "i386-unknown-unknown"' %t.log
// RUN: grep '"-S"' %t.log
// RUN: grep '"-disable-free"' %t.log
// RUN: grep '"-mrelocation-model" "static"' %t.log
// RUN: grep '"-mdisable-fp-elim"' %t.log
// RUN: grep '"-munwind-tables"' %t.log
// RUN: grep '"-Os"' %t.log
// RUN: grep '"-o" .*clang-translation.*' %t.log
// RUN: grep '"-masm-verbose"' %t.log
// RUN: clang -ccc-host-triple i386-apple-darwin9 -### -S %s -o %t.s 2> %t.log
// RUN: grep '"-mcpu" "yonah"' %t.log
// RUN: clang -ccc-host-triple x86_64-apple-darwin9 -### -S %s -o %t.s 2> %t.log
// RUN: grep '"-mcpu" "core2"' %t.log
