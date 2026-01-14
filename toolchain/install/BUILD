# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load(
    "@llvm-project//:vars.bzl",
    "LLVM_VERSION_MAJOR",
)
load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("//bazel/cc_rules:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//bazel/manifest:defs.bzl", "manifest")
load("//toolchain/base:llvm_tools.bzl", "LLVM_MAIN_TOOLS", "LLVM_TOOL_ALIASES")
load("//toolchain/base:runtime_sources.bzl", "BUILTINS_FILEGROUPS", "CRT_FILES")
load("configure_cmake_file.bzl", "configure_cmake_file")
load("install_filegroups.bzl", "install_filegroup", "install_symlink", "install_target", "make_install_filegroups")
load("pkg_helpers.bzl", "pkg_naming_variables", "pkg_tar_and_test")

package(default_visibility = ["//visibility:public"])

# Build rules supporting the install data tree for the Carbon toolchain.
#
# This populates a synthetic Carbon toolchain installation under the
# `prefix` directory. For details on its layout, see `install_dirs` below.

cc_library(
    name = "busybox_info",
    srcs = ["busybox_info.cpp"],
    hdrs = ["busybox_info.h"],
    deps = [
        "//common:error",
        "//common:exe_path",
        "//common:filesystem",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "busybox_info_test",
    size = "small",
    srcs = ["busybox_info_test.cpp"],
    deps = [
        ":busybox_info",
        "//common:check",
        "//common:filesystem",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

# This target doesn't include prelude libraries. To get a target that has the
# prelude available, use //toolchain.
cc_binary(
    name = "carbon-busybox",
    srcs = ["busybox_main.cpp"],
    deps = [
        ":busybox_info",
        "//common:all_llvm_targets",
        "//common:bazel_working_dir",
        "//common:error",
        "//common:exe_path",
        "//common:init_llvm",
        "//common:version_stamp",
        "//toolchain/base:install_paths",
        "//toolchain/base:llvm_tools_def",
        "//toolchain/driver",
        "@llvm-project//llvm:Support",
    ],
)

clang_aliases = [
    "clang",
    "clang++",
    "clang-cl",
    "clang-cpp",
]

# TODO: Add remaining aliases of LLD for Windows and WASM when we have support
# for them wired up through the busybox.
lld_aliases = [
    "ld.lld",
    "ld64.lld",
]

llvm_binaries = clang_aliases + lld_aliases + [
    tool.bin_name
    for tool in LLVM_MAIN_TOOLS.values()
] + [
    "llvm-" + alias
    for (_, aliases) in LLVM_TOOL_ALIASES.items()
    for alias in aliases
]

filegroup(
    name = "clang_headers",
    srcs = ["@llvm-project//clang:builtin_headers_gen"],
)

# Collect the runtime sources that are collectively installed into the
# `builtins` directory.
filegroup(
    name = "clang_builtins_runtimes",
    srcs = CRT_FILES.values() + BUILTINS_FILEGROUPS.values(),
)

py_binary(
    name = "configure_cmake_file_impl",
    srcs = ["configure_cmake_file_impl.py"],
)

configure_cmake_file(
    name = "libcxx_site_config_gen",
    src = "@llvm-project//libcxx:include/__config_site.in",
    out = "staging_libcxx/include/__config_site",
    defines = {
        # We can inject custom logic at the end of the site configuration with
        # the ABI defines. This can custom set (or re-set) any of the relevant
        # configuration defines. Note that while this is sorted here in the
        # BUILD file, it is expanded at the _end_ of the configuration header
        # and so overrides the other configuration settings.
        #
        # TODO: This is a lot of C++ code to embed into a BUILD file. Even
        # though it moves it farther from the interacting CMake defines, we
        # should look at factoring this into a header that is included.
        "_LIBCPP_ABI_DEFINES": "\n".join([
            # We want to install a single header that works in all build modes,
            # so we define the ABI namespace based on how the header is used
            # rather than a fixed one. However, we only support use with Clang
            # and so we assume `__has_feature` is available and works.
            #
            # Note that generally, we don't rely on different ABI namespaces for
            # functionality -- the distinction is more to make errors when
            # linking with the wrong build of the standard library obvious and
            # immediate. We only can achieve this for sanitizers that have a
            # preprocessor detectable model.
            "#if __has_feature(address_sanitizer)",
            "# undef _LIBCPP_ABI_NAMESPACE",
            "# define _LIBCPP_ABI_NAMESPACE __asan",
            # Also mark that libc++ will be instrumented.
            "# undef _LIBCPP_INSTRUMENTED_WITH_ASAN",
            "# define _LIBCPP_INSTRUMENTED_WITH_ASAN 1",
            "#elif __has_feature(memory_sanitizer)",
            # TODO: If a track-origins macro becomes available, we should
            # distinguish that case, too.
            "# undef _LIBCPP_ABI_NAMESPACE",
            "# define _LIBCPP_ABI_NAMESPACE __msan",
            "#elif __has_feature(thread_sanitizer)",
            "# undef _LIBCPP_ABI_NAMESPACE",
            "# define _LIBCPP_ABI_NAMESPACE __tsan",
            "#elif __has_feature(cfi_sanitizer)",
            "# undef _LIBCPP_ABI_NAMESPACE",
            "# define _LIBCPP_ABI_NAMESPACE __cfi",
            "#endif",
            "",

            # Establish a default hardening mode where possible.
            "#ifndef _LIBCPP_HARDENING_MODE",
            "# ifndef NDEBUG",
            # !NDEBUG has significant overhead anyway and is explicitly a
            # debugging build rather than a production build.
            "#  define _LIBCPP_HARDENING_MODE _LIBCPP_HARDENING_MODE_DEBUG",
            "# else",
            # Default to the fast hardening checks.
            "#  define _LIBCPP_HARDENING_MODE _LIBCPP_HARDENING_MODE_FAST",
            "# endif",
            "#endif",
            "",

            # CUDA can't call any existing abort implementations, so disable
            # hardening there.
            "#ifdef __CUDA__",
            "# undef _LIBCPP_HARDENING_MODE",
            "# define _LIBCPP_HARDENING_MODE _LIBCPP_HARDENING_MODE_NONE",
            "#endif",
            "",

            # Setup platform-dependent features using preprocessor logic.
            "#ifdef __linux__",
            "# undef _LIBCPP_HAS_TIME_ZONE_DATABASE",
            "# define _LIBCPP_HAS_TIME_ZONE_DATABASE 1",
            "#endif",
            "",

            # Mixing translation units compiled with different versions of
            # libc++ is unsupported. Disable ABI tags to decrease symbol
            # lengths.
            "#define _LIBCPP_NO_ABI_TAG",
        ]),

        # No forced ABI.
        "_LIBCPP_ABI_FORCE_ITANIUM": "OFF",
        "_LIBCPP_ABI_FORCE_MICROSOFT": "OFF",

        # We use the unstable ABI and define a custom, Carbon-specific ABI
        # namespace. This also matches the mangling prefix used for Carbon
        # symbols.
        "_LIBCPP_ABI_NAMESPACE": "_C",
        # TODO: Fix the need to define _LIBCPP_ABI_VERSION when the unstable
        # ABI is selected.
        "_LIBCPP_ABI_VERSION": "999",

        # Follow hardening mode for the assertion semantics.
        "_LIBCPP_ASSERTION_SEMANTIC_DEFAULT": "_LIBCPP_ASSERTION_SEMANTIC_HARDENING_DEPENDENT",

        # Enable various features in libc++ available across platforms. We
        # describe these in a block to allow the BUILD file to sort them.
        #
        # - We enable threads, and use auto-detection rather than forcing an
        #   API.
        # - Availability annotations do not apply to Carbon's libc++, so those
        #   are disabled.
        #
        # Where there are platform differences in the features, we disable them
        # here and re-enable them in the `_LIBCPP_ABI_DEFINES` section using
        # custom logic to detect the relevant platform.
        "_LIBCPP_HAS_FILESYSTEM": "ON",
        "_LIBCPP_HAS_LOCALIZATION": "ON",
        "_LIBCPP_HAS_MONOTONIC_CLOCK": "ON",
        "_LIBCPP_HAS_NO_VENDOR_AVAILABILITY_ANNOTATIONS": "ON",
        "_LIBCPP_HAS_RANDOM_DEVICE": "ON",
        "_LIBCPP_HAS_TERMINAL": "ON",
        "_LIBCPP_HAS_THREADS": "ON",
        "_LIBCPP_HAS_THREAD_API_EXTERNAL": "OFF",
        "_LIBCPP_HAS_THREAD_API_PTHREAD": "OFF",
        "_LIBCPP_HAS_THREAD_API_WIN32": "OFF",
        "_LIBCPP_HAS_TIME_ZONE_DATABASE": "OFF",
        "_LIBCPP_HAS_UNICODE": "ON",
        "_LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS": "OFF",
        "_LIBCPP_HAS_WIDE_CHARACTERS": "ON",

        # When ASan is enabled, we ensure that libc++ is built with it as well.
        # However, we can't set this more carefully here so we set it to off and
        # override it below when using ASan.
        "_LIBCPP_INSTRUMENTED_WITH_ASAN": "OFF",

        # Set the parallel backend to serial.
        # TODO: We should revisit this.
        "_LIBCPP_PSTL_BACKEND_SERIAL": "1",
    },
)

configure_cmake_file(
    name = "libcxx_assertion_handler_gen",
    src = "@llvm-project//libcxx:vendor/llvm/default_assertion_handler.in",
    out = "staging_libcxx/include/__assertion_handler",
    defines = {
        # Currently the default handler needs no substitutions.
    },
)

filegroup(
    name = "libcxx",
    srcs = [
        "@llvm-project//libcxx:libcxx_hdrs",
        "@llvm-project//libcxx:libcxx_srcs",
    ],
)

filegroup(
    name = "libcxx_gen_files",
    srcs = [
        "staging_libcxx/include/__assertion_handler",
        "staging_libcxx/include/__config_site",
    ],
)

filegroup(
    name = "libcxxabi",
    srcs = [
        "@llvm-project//libcxxabi:libcxxabi_hdrs",
        "@llvm-project//libcxxabi:libcxxabi_srcs",
    ],
)

filegroup(
    name = "libunwind",
    srcs = [
        "@llvm-project//libunwind:libunwind_hdrs",
        "@llvm-project//libunwind:libunwind_srcs",
    ],
)

# Currently, we're only installing the subset of LLVM's libc internals needed to
# build libc++. At some point, we should ship LLVM's libc itself, and that will
# likely expand this to cover more of the source. However, we'll still want to
# distinguish between the _internal_ installation and the generated set of
# headers that we inject into the include search for user compiles. The
# `include` subdirectory in this file group is _not_ intended to be exposed to
# user compiles, only to compilation of runtimes.
filegroup(
    name = "libc_internal",
    srcs = [
        "@llvm-project//libc:libcxx_shared_headers_hdrs",
    ],
)

# Given a CMake-style install prefix[1], the hierarchy looks like:
#
# - prefix/bin: Binaries intended for direct use.
# - prefix/lib/carbon: Private data and files.
# - prefix/lib/carbon/core: The `Core` package files.
# - prefix/lib/carbon/llvm/bin: LLVM binaries.
#
# This will be how installs are provided on Unix-y platforms, and is loosely
# based on the FHS (Filesystem Hierarchy Standard). See the CMake install prefix
# documentation[1] for more details.
#
# [1]: https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html
install_dirs = {
    "bin": [
        install_symlink(
            "carbon",
            "../lib/carbon/carbon-busybox",
            is_driver = True,
        ),
    ],
    "lib/carbon": [
        install_target("carbon_install.txt", "carbon_install.txt"),
        install_target(
            "install_digest.txt",
            ":install_digest.txt",
            executable = False,
            is_digest = True,
            is_driver = True,
        ),
        install_target(
            "carbon-busybox",
            ":carbon-busybox",
            executable = True,
            is_driver = True,
        ),
        # TODO: Consider if we want to keep `core` here or group it with
        # runtimes. It is a bit of both -- standard library, and runtimes.
        install_filegroup("core", "//core:prelude"),
    ],
    "lib/carbon/llvm/bin": [install_symlink(
        name,
        "../../carbon-busybox",
        is_driver = True,
    ) for name in llvm_binaries],
    "lib/carbon/runtimes": [
        install_filegroup(
            "builtins",
            ":clang_builtins_runtimes",
            remove_prefix = "lib/builtins/",
        ),
        install_filegroup("libcxx", ":libcxx"),
        install_filegroup("libcxxabi", ":libcxxabi"),
        install_filegroup("libunwind", ":libunwind"),
    ],
    "lib/carbon/runtimes/libc": [
        install_filegroup("internal", ":libc_internal"),
    ],
    "lib/carbon/runtimes/libcxx": [
        install_filegroup(
            "include",
            ":libcxx_gen_files",
            remove_prefix = "staging_libcxx/include/",
        ),
    ],
    "lib/carbon/llvm/lib/clang/" + LLVM_VERSION_MAJOR: [
        install_filegroup(
            "include",
            ":clang_headers",
            label = "installed_clang_headers",
            remove_prefix = "staging/include/",
        ),
    ],
    "lib/carbon/llvm/lib/clang/" + LLVM_VERSION_MAJOR + "/include": [
        install_filegroup(
            "fuzzer",
            "@llvm-project//compiler-rt:fuzzer_installed_hdrs",
            remove_prefix = "include/fuzzer/",
        ),
        install_filegroup(
            "profile",
            "@llvm-project//compiler-rt:profile_installed_hdrs",
            remove_prefix = "include/profile/",
        ),
        install_filegroup(
            "sanitizer",
            "@llvm-project//compiler-rt:sanitizer_installed_hdrs",
            remove_prefix = "include/sanitizer/",
        ),
    ],
}

make_install_filegroups(
    name = "install_data",
    install_dirs = install_dirs,
    no_digest_name = "install_data.no_digest",
    no_driver_name = "install_data.no_driver",
    pkg_name = "pkg_data",
    prefix = "prefix",
)

py_test(
    name = "llvm_symlinks_test",
    size = "small",
    srcs = ["llvm_symlinks_test.py"],
    data = [
        ":install_data",
        "//toolchain/driver:prebuilt_runtimes",
    ],
    deps = ["@bazel_tools//tools/python/runfiles"],
)

manifest(
    name = "install_data_manifest.txt",
    srcs = [":install_data"],
)

cc_binary(
    name = "make-installation-digest",
    srcs = ["make_installation_digest.cpp"],
    deps = [
        "//common:bazel_working_dir",
        "//common:error",
        "//common:exe_path",
        "//common:filesystem",
        "//common:init_llvm",
        "//common:map",
        "//common:vlog",
        "@llvm-project//llvm:Support",
    ],
)

manifest(
    name = "install_digest_manifest.txt",
    srcs = [":install_data.no_digest"],
)

genrule(
    name = "gen_digest",
    srcs = [
        ":install_data.no_digest",
        "install_digest_manifest.txt",
    ],
    outs = [":install_digest.txt"],
    cmd = "$(location :make-installation-digest) " +
          "$(location install_digest_manifest.txt) $@",
    tools = [":make-installation-digest"],
)

# A list of clang's installed builtin header files.
# This is consumed by //toolchain/testing:file_test.
manifest(
    name = "clang_headers_manifest.txt",
    srcs = [":installed_clang_headers"],
    strip_package_dir = True,
)

pkg_naming_variables(
    name = "packaging_variables",
)

# We build both a compressed and uncompressed tar file with the same code here.
# This lets us use the tar file in testing as it is fast to create, but ship the
# compressed version as a release.
#
# For manual tests, the tar rules are `carbon_toolchain_tar_rule` and
# `carbon_toolchain_tar_gz_rule`.
pkg_tar_and_test(
    srcs = [":pkg_data"],
    install_data_manifest = ":install_data_manifest.txt",
    name_base = "carbon_toolchain",
    package_dir = "carbon_toolchain-$(version)",
    package_file_name_base = "carbon_toolchain-$(version)",
    package_variables = ":packaging_variables",
    stamp = -1,  # Allow `--stamp` builds to produce file timestamps.
)
