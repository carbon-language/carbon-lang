#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

from libcxx.test.dsl import *
import re
import shutil
import sys
import subprocess

_isClang      = lambda cfg: '__clang__' in compilerMacros(cfg) and '__apple_build_version__' not in compilerMacros(cfg)
_isAppleClang = lambda cfg: '__apple_build_version__' in compilerMacros(cfg)
_isGCC        = lambda cfg: '__GNUC__' in compilerMacros(cfg) and '__clang__' not in compilerMacros(cfg)
_isMSVC       = lambda cfg: '_MSC_VER' in compilerMacros(cfg)
_msvcVersion  = lambda cfg: (int(compilerMacros(cfg)['_MSC_VER']) // 100, int(compilerMacros(cfg)['_MSC_VER']) % 100)

DEFAULT_FEATURES = [
  Feature(name='fcoroutines-ts',
          when=lambda cfg: hasCompileFlag(cfg, '-fcoroutines-ts') and
                           featureTestMacros(cfg, flags='-fcoroutines-ts').get('__cpp_coroutines', 0) >= 201703,
          actions=[AddCompileFlag('-fcoroutines-ts')]),

  Feature(name='thread-safety',
          when=lambda cfg: hasCompileFlag(cfg, '-Werror=thread-safety'),
          actions=[AddCompileFlag('-Werror=thread-safety')]),

  Feature(name='diagnose-if-support',
          when=lambda cfg: hasCompileFlag(cfg, '-Wuser-defined-warnings'),
          actions=[AddCompileFlag('-Wuser-defined-warnings')]),

  Feature(name='has-fblocks',                   when=lambda cfg: hasCompileFlag(cfg, '-fblocks')),
  Feature(name='-fsized-deallocation',          when=lambda cfg: hasCompileFlag(cfg, '-fsized-deallocation')),
  Feature(name='-faligned-allocation',          when=lambda cfg: hasCompileFlag(cfg, '-faligned-allocation')),
  Feature(name='fdelayed-template-parsing',     when=lambda cfg: hasCompileFlag(cfg, '-fdelayed-template-parsing')),
  Feature(name='libcpp-no-concepts',            when=lambda cfg: featureTestMacros(cfg).get('__cpp_concepts', 0) < 201907),
  Feature(name='libcpp-no-coroutines',          when=lambda cfg: featureTestMacros(cfg).get('__cpp_impl_coroutine', 0) < 201902),
  Feature(name='has-fobjc-arc',                 when=lambda cfg: hasCompileFlag(cfg, '-xobjective-c++ -fobjc-arc') and
                                                                 sys.platform.lower().strip() == 'darwin'), # TODO: this doesn't handle cross-compiling to Apple platforms.
  Feature(name='objective-c++',                 when=lambda cfg: hasCompileFlag(cfg, '-xobjective-c++ -fobjc-arc')),
  Feature(name='verify-support',                when=lambda cfg: hasCompileFlag(cfg, '-Xclang -verify-ignore-unexpected')),

  Feature(name='non-lockfree-atomics',
          when=lambda cfg: sourceBuilds(cfg, """
            #include <atomic>
            struct Large { int storage[100]; };
            std::atomic<Large> x;
            int main(int, char**) { (void)x.load(); return 0; }
          """)),
  # TODO: Remove this feature once compiler-rt includes __atomic_is_lockfree()
  # on all supported platforms.
  Feature(name='is-lockfree-runtime-function',
          when=lambda cfg: sourceBuilds(cfg, """
            #include <atomic>
            struct Large { int storage[100]; };
            std::atomic<Large> x;
            int main(int, char**) { return x.is_lock_free(); }
          """)),

  # Some tests rely on creating shared libraries which link in the C++ Standard Library. In some
  # cases, this doesn't work (e.g. if the library was built as a static archive and wasn't compiled
  # as position independent). This feature informs the test suite of whether it's possible to create
  # a shared library in a shell test by using the '-shared' compiler flag.
  #
  # Note: To implement this check properly, we need to make sure that we use something inside the
  # compiled library, not only in the headers. It should be safe to assume that all implementations
  # define `operator new` in the compiled library.
  Feature(name='cant-build-shared-library',
          when=lambda cfg: not sourceBuilds(cfg, """
            void f() { new int(3); }
          """, ['-shared'])),

  # Whether Bash can run on the executor.
  # This is not always the case, for example when running on embedded systems.
  #
  # For the corner case of bash existing, but it being missing in the path
  # set in %{exec} as "--env PATH=one-single-dir", the executor does find
  # and executes bash, but bash then can't find any other common shell
  # utilities. Test executing "bash -c 'bash --version'" to see if bash
  # manages to find binaries to execute.
  Feature(name='executor-has-no-bash',
          when=lambda cfg: runScriptExitCode(cfg, ['%{exec} bash -c \'bash --version\'']) != 0),

  Feature(name='apple-clang',                                                                                                      when=_isAppleClang),
  Feature(name=lambda cfg: 'apple-clang-{__clang_major__}'.format(**compilerMacros(cfg)),                                          when=_isAppleClang),
  Feature(name=lambda cfg: 'apple-clang-{__clang_major__}.{__clang_minor__}'.format(**compilerMacros(cfg)),                        when=_isAppleClang),
  Feature(name=lambda cfg: 'apple-clang-{__clang_major__}.{__clang_minor__}.{__clang_patchlevel__}'.format(**compilerMacros(cfg)), when=_isAppleClang),

  Feature(name='clang',                                                                                                            when=_isClang,
          actions=[AddCompileFlag('-D_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER')]),
  Feature(name=lambda cfg: 'clang-{__clang_major__}'.format(**compilerMacros(cfg)),                                                when=_isClang),
  Feature(name=lambda cfg: 'clang-{__clang_major__}.{__clang_minor__}'.format(**compilerMacros(cfg)),                              when=_isClang),
  Feature(name=lambda cfg: 'clang-{__clang_major__}.{__clang_minor__}.{__clang_patchlevel__}'.format(**compilerMacros(cfg)),       when=_isClang),

  Feature(name='gcc',                                                                                                              when=_isGCC),
  Feature(name=lambda cfg: 'gcc-{__GNUC__}'.format(**compilerMacros(cfg)),                                                         when=_isGCC),
  Feature(name=lambda cfg: 'gcc-{__GNUC__}.{__GNUC_MINOR__}'.format(**compilerMacros(cfg)),                                        when=_isGCC),
  Feature(name=lambda cfg: 'gcc-{__GNUC__}.{__GNUC_MINOR__}.{__GNUC_PATCHLEVEL__}'.format(**compilerMacros(cfg)),                  when=_isGCC),

  Feature(name='msvc',                                                                                                             when=_isMSVC),
  Feature(name=lambda cfg: 'msvc-{}'.format(*_msvcVersion(cfg)),                                                                   when=_isMSVC),
  Feature(name=lambda cfg: 'msvc-{}.{}'.format(*_msvcVersion(cfg)),                                                                when=_isMSVC),
]

# Deduce and add the test features that that are implied by the #defines in
# the <__config_site> header.
#
# For each macro of the form `_LIBCPP_XXX_YYY_ZZZ` defined below that
# is defined after including <__config_site>, add a Lit feature called
# `libcpp-xxx-yyy-zzz`. When a macro is defined to a specific value
# (e.g. `_LIBCPP_ABI_VERSION=2`), the feature is `libcpp-xxx-yyy-zzz=<value>`.
macros = {
  '_LIBCPP_HAS_NO_MONOTONIC_CLOCK': 'libcpp-has-no-monotonic-clock',
  '_LIBCPP_HAS_NO_THREADS': 'libcpp-has-no-threads',
  '_LIBCPP_HAS_THREAD_API_EXTERNAL': 'libcpp-has-thread-api-external',
  '_LIBCPP_HAS_THREAD_API_PTHREAD': 'libcpp-has-thread-api-pthread',
  '_LIBCPP_NO_VCRUNTIME': 'libcpp-no-vcruntime',
  '_LIBCPP_ABI_VERSION': 'libcpp-abi-version',
  '_LIBCPP_ABI_UNSTABLE': 'libcpp-abi-unstable',
  '_LIBCPP_HAS_NO_FILESYSTEM_LIBRARY': 'libcpp-has-no-filesystem-library',
  '_LIBCPP_HAS_NO_RANDOM_DEVICE': 'libcpp-has-no-random-device',
  '_LIBCPP_HAS_NO_LOCALIZATION': 'libcpp-has-no-localization',
  '_LIBCPP_HAS_NO_WIDE_CHARACTERS': 'libcpp-has-no-wide-characters',
  '_LIBCPP_HAS_NO_INCOMPLETE_FORMAT': 'libcpp-has-no-incomplete-format',
  '_LIBCPP_HAS_NO_INCOMPLETE_RANGES': 'libcpp-has-no-incomplete-ranges',
  '_LIBCPP_HAS_NO_UNICODE': 'libcpp-has-no-unicode',
}
for macro, feature in macros.items():
  DEFAULT_FEATURES += [
    Feature(name=lambda cfg, m=macro, f=feature: f + (
              '={}'.format(compilerMacros(cfg)[m]) if compilerMacros(cfg)[m] else ''
            ),
            when=lambda cfg, m=macro: m in compilerMacros(cfg),

            # FIXME: This is a hack that should be fixed using module maps.
            # If modules are enabled then we have to lift all of the definitions
            # in <__config_site> onto the command line.
            actions=lambda cfg, m=macro: [
              AddCompileFlag('-Wno-macro-redefined -D{}'.format(m) + (
                '={}'.format(compilerMacros(cfg)[m]) if compilerMacros(cfg)[m] else ''
              ))
            ]
    )
  ]


# Mapping from canonical locale names (used in the tests) to possible locale
# names on various systems. Each locale is considered supported if any of the
# alternative names is supported.
locales = {
  'en_US.UTF-8':     ['en_US.UTF-8', 'en_US.utf8', 'English_United States.1252'],
  'fr_FR.UTF-8':     ['fr_FR.UTF-8', 'fr_FR.utf8', 'French_France.1252'],
  'ru_RU.UTF-8':     ['ru_RU.UTF-8', 'ru_RU.utf8', 'Russian_Russia.1251'],
  'zh_CN.UTF-8':     ['zh_CN.UTF-8', 'zh_CN.utf8', 'Chinese_China.936'],
  'fr_CA.ISO8859-1': ['fr_CA.ISO8859-1', 'French_Canada.1252'],
  'cs_CZ.ISO8859-2': ['cs_CZ.ISO8859-2', 'Czech_Czech Republic.1250']
}
for locale, alts in locales.items():
  # Note: Using alts directly in the lambda body here will bind it to the value at the
  # end of the loop. Assigning it to a default argument works around this issue.
  DEFAULT_FEATURES.append(Feature(name='locale.{}'.format(locale),
                                  when=lambda cfg, alts=alts: hasAnyLocale(cfg, alts)))


# Add features representing the platform name: darwin, linux, windows, etc...
DEFAULT_FEATURES += [
  Feature(name='darwin', when=lambda cfg: '__APPLE__' in compilerMacros(cfg)),
  Feature(name='windows', when=lambda cfg: '_WIN32' in compilerMacros(cfg)),
  Feature(name='windows-dll', when=lambda cfg: '_WIN32' in compilerMacros(cfg) and not '_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS' in compilerMacros(cfg)),
  Feature(name='linux', when=lambda cfg: '__linux__' in compilerMacros(cfg)),
  Feature(name='netbsd', when=lambda cfg: '__NetBSD__' in compilerMacros(cfg)),
  Feature(name='freebsd', when=lambda cfg: '__FreeBSD__' in compilerMacros(cfg))
]

# Add features representing the build host platform name.
# The build host could differ from the target platform for cross-compilation.
DEFAULT_FEATURES += [
  Feature(name='buildhost={}'.format(sys.platform.lower().strip())),
  # sys.platform can be represented by "sub-system" on Windows host, such as 'win32', 'cygwin', 'mingw' & etc.
  # Here is a consolidated feature for the build host plaform name on Windows.
  Feature(name='buildhost=windows', when=lambda cfg: platform.system().lower().startswith('windows'))
]

# Detect whether GDB is on the system, has Python scripting and supports
# adding breakpoint commands. If so add a substitution to access it.
def check_gdb(cfg):
  gdb_path = shutil.which('gdb')
  if gdb_path is None:
    return False

  # Check that we can set breakpoint commands, which was added in 8.3.
  # Using the quit command here means that gdb itself exits, not just
  # the "python <...>" command.
  test_src = """\
try:
  gdb.Breakpoint(\"main\").commands=\"foo\"
except AttributeError:
  gdb.execute(\"quit 1\")
gdb.execute(\"quit\")"""

  try:
    stdout = subprocess.check_output(
              [gdb_path, "-ex", "python " + test_src, "--batch"],
              stderr=subprocess.DEVNULL, universal_newlines=True)
  except subprocess.CalledProcessError:
    # We can't set breakpoint commands
    return False

  # Check we actually ran the Python
  return not "Python scripting is not supported" in stdout

DEFAULT_FEATURES += [
  Feature(name='host-has-gdb-with-python',
    when=check_gdb,
    actions=[AddSubstitution('%{gdb}', lambda cfg: shutil.which('gdb'))]
  )
]
