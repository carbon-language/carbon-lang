// REQUIRES: crash-recovery, shell, system-darwin

// RUN: rm -rf %t
// RUN: mkdir -p %t/m %t/i/Foo.framework/Headers
// RUN: echo '// Foo.h' > %t/i/Foo.framework/Headers/Foo.h
// RUN: %hmaptool write %S/../Preprocessor/Inputs/headermap-rel/foo.hmap.json %t/i/foo.hmap

// RUN: env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: not %clang -fsyntax-only -fmodules -fmodules-cache-path=%t/m %s \
// RUN:     -I %t/i/foo.hmap -F %t/i 2>&1 | FileCheck %s

// RUN: FileCheck --check-prefix=CHECKSH %s -input-file %t/crash-vfs-*.sh
// RUN: FileCheck --check-prefix=CHECKYAML %s -input-file \
// RUN:   %t/crash-vfs-*.cache/vfs/vfs.yaml

#include "Foo.h"
#include "Foo.h"

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.m
// CHECK-NEXT: note: diagnostic msg: {{.*}}.cache

// CHECKSH: # Crash reproducer
// CHECKSH-NEXT: # Driver args: "-fsyntax-only"
// CHECKSH-NEXT: # Original command: {{.*$}}
// CHECKSH-NEXT: "-cc1"
// CHECKSH: "-I" "/[[INCPATH:.*]]/foo.hmap"
// CHECKSH: "crash-vfs-{{[^ ]*}}.m"
// CHECKSH: "-ivfsoverlay" "crash-vfs-{{[^ ]*}}.cache/vfs/vfs.yaml"
// CHECKSH: "-fmodules-cache-path=crash-vfs-{{[^ ]*}}.cache/repro-modules"

// CHECKYAML: 'case-sensitive':
// CHECKYAML-NEXT: 'use-external-names': 'false',
// CHECKYAML-NEXT: 'overlay-relative': 'true',
// CHECKYAML: 'type': 'directory'
// CHECKYAML: 'name': "/[[PATH:.*]]/Foo.framework/Headers",
// CHECKYAML-NEXT: 'contents': [
// CHECKYAML-NEXT:   {
// CHECKYAML-NEXT:     'type': 'file',
// CHECKYAML-NEXT:     'name': "Foo.h",
// CHECKYAML-NEXT:     'external-contents': "/[[PATH]]/Foo.framework/Headers/Foo.h"
// CHECKYAML: 'type': 'directory'
// CHECKYAML: 'name': "/[[PATH:.*]]/i",
// CHECKYAML-NEXT: 'contents': [
// CHECKYAML-NEXT:   {
// CHECKYAML-NEXT:     'type': 'file',
// CHECKYAML-NEXT:     'name': "foo.hmap",
// CHECKYAML-NEXT:     'external-contents': "/[[PATH]]/i/foo.hmap"
