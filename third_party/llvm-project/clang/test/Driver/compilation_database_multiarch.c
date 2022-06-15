// REQUIRES: system-darwin

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -fdriver-only -o %t/out %s -mtargetos=macos12 -arch arm64 -arch x86_64 -MJ %t/compilation_database.json
// RUN: FileCheck --input-file=%t/compilation_database.json %s

// CHECK:      { "directory": "{{.*}}", "file": "{{.*}}", "output": "[[OUTPUT_X86_64:.*]]", "arguments": [{{.*}}, "-o", "[[OUTPUT_X86_64]]", {{.*}} "--target=x86_64-apple-macosx12.0.0"]},
// CHECK-NEXT: { "directory": "{{.*}}", "file": "{{.*}}", "output": "[[OUTPUT_ARM64:.*]]", "arguments": [{{.*}}, "-o", "[[OUTPUT_ARM64]]", {{.*}} "--target=arm64-apple-macosx12.0.0"]},
