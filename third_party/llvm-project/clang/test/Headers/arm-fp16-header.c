// RUN: %clang -fsyntax-only  -ffreestanding              --target=aarch64-none-eabi -march=armv8.2-a+fp16 -std=c89 -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64-none-eabi -march=armv8.2-a+fp16 -std=c99 -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64-none-eabi -march=armv8.2-a+fp16 -std=c11 -xc %s

// RUN: %clang -fsyntax-only -ffreestanding               --target=aarch64_be-none-eabi -march=armv8.2-a+fp16 -std=c89 -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64_be-none-eabi -march=armv8.2-a+fp16 -std=c99 -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64_be-none-eabi -march=armv8.2-a+fp16 -std=c11 -xc %s

// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-eabi -march=armv8.2-a+fp16 -std=c++98 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-eabi -march=armv8.2-a+fp16 -std=c++11 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-eabi -march=armv8.2-a+fp16 -std=c++14 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-eabi -march=armv8.2-a+fp16 -std=c++17 -xc++ %s

// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-eabi -march=armv8.2-a+fp16 -std=c++98 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-eabi -march=armv8.2-a+fp16 -std=c++11 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-eabi -march=armv8.2-a+fp16 -std=c++14 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-eabi -march=armv8.2-a+fp16 -std=c++17 -xc++ %s

#include <arm_fp16.h>
