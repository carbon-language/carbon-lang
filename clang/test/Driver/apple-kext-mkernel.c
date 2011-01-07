// RUN: %clang -ccc-host-triple x86_64-apple-darwin10 -mkernel -### -fsyntax-only %s 2> %t
// RUN grep "-disable-red-zone" %t
// RUN grep "-fapple-kext" %t
// RUN grep "-fno-builtin" %t
// RUN grep "-fno-rtti" %t
// RUN grep "-fno-common" %t
