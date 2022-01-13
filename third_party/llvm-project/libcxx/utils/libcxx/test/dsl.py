#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

import os
import pickle
import pipes
import platform
import re
import shutil
import tempfile

import libcxx.test.format
import lit
import lit.LitConfig
import lit.Test
import lit.TestRunner
import lit.util

class ConfigurationError(Exception):
  pass

class ConfigurationCompilationError(ConfigurationError):
  pass

class ConfigurationRuntimeError(ConfigurationError):
  pass

def _memoizeExpensiveOperation(extractCacheKey):
  """
  Allows memoizing a very expensive operation.

  We pickle the cache key to make sure we store an immutable representation
  of it. If we stored an object and the object was referenced elsewhere, it
  could be changed from under our feet, which would break the cache.
  """
  def decorator(function):
    cache = {}
    def f(*args, **kwargs):
      cacheKey = pickle.dumps(extractCacheKey(*args, **kwargs))
      if cacheKey not in cache:
        cache[cacheKey] = function(*args, **kwargs)
      return cache[cacheKey]
    return f
  return decorator

def _executeScriptInternal(test, commands):
  """
  Returns (stdout, stderr, exitCode, timeoutInfo)

  TODO: This really should be easier to access from Lit itself
  """
  parsedCommands = libcxx.test.format.parseScript(test, preamble=commands)

  litConfig = lit.LitConfig.LitConfig(
    progname='lit',
    path=[],
    quiet=False,
    useValgrind=False,
    valgrindLeakCheck=False,
    valgrindArgs=[],
    noExecute=False,
    debug=False,
    isWindows=platform.system() == 'Windows',
    params={})
  _, tmpBase = libcxx.test.format._getTempPaths(test)
  execDir = os.path.dirname(test.getExecPath())
  res = lit.TestRunner.executeScriptInternal(test, litConfig, tmpBase, parsedCommands, execDir)
  if isinstance(res, lit.Test.Result): # Handle failure to parse the Lit test
    res = ('', res.output, 127, None)
  (out, err, exitCode, timeoutInfo) = res

  # TODO: As a temporary workaround until https://reviews.llvm.org/D81892 lands, manually
  #       split any stderr output that is included in stdout. It shouldn't be there, but
  #       the Lit internal shell conflates stderr and stdout.
  conflatedErrorOutput = re.search("(# command stderr:.+$)", out, flags=re.DOTALL)
  if conflatedErrorOutput:
    conflatedErrorOutput = conflatedErrorOutput.group(0)
    out = out[:-len(conflatedErrorOutput)]
    err += conflatedErrorOutput

  return (out, err, exitCode, timeoutInfo)

def _makeConfigTest(config):
  # Make sure the support directories exist, which is needed to create
  # the temporary file %t below.
  sourceRoot = os.path.join(config.test_exec_root, '__config_src__')
  execRoot = os.path.join(config.test_exec_root, '__config_exec__')
  for supportDir in (sourceRoot, execRoot):
    if not os.path.exists(supportDir):
      os.makedirs(supportDir)

  # Create a dummy test suite and single dummy test inside it. As part of
  # the Lit configuration, automatically do the equivalent of 'mkdir %T'
  # and 'rm -r %T' to avoid cluttering the build directory.
  suite = lit.Test.TestSuite('__config__', sourceRoot, execRoot, config)
  tmp = tempfile.NamedTemporaryFile(dir=sourceRoot, delete=False, suffix='.cpp')
  tmp.close()
  pathInSuite = [os.path.relpath(tmp.name, sourceRoot)]
  class TestWrapper(lit.Test.Test):
    def __enter__(self):
      testDir, _ = libcxx.test.format._getTempPaths(self)
      os.makedirs(testDir)
      return self
    def __exit__(self, *args):
      testDir, _ = libcxx.test.format._getTempPaths(self)
      shutil.rmtree(testDir)
      os.remove(tmp.name)
  return TestWrapper(suite, pathInSuite, config)

@_memoizeExpensiveOperation(lambda c, s, f=[]: (c.substitutions, c.environment, s, f))
def sourceBuilds(config, source, additionalFlags=[]):
  """
  Return whether the program in the given string builds successfully.

  This is done by compiling and linking a program that consists of the given
  source with the %{cxx} substitution, and seeing whether that succeeds. If
  any additional flags are passed, they are appended to the compiler invocation.
  """
  with _makeConfigTest(config) as test:
    with open(test.getSourcePath(), 'w') as sourceFile:
      sourceFile.write(source)
    _, _, exitCode, _ = _executeScriptInternal(test, ['%{{build}} {}'.format(' '.join(additionalFlags))])
    return exitCode == 0

@_memoizeExpensiveOperation(lambda c, p, args=None: (c.substitutions, c.environment, p, args))
def programOutput(config, program, args=None):
  """
  Compiles a program for the test target, run it on the test target and return
  the output.

  Note that execution of the program is done through the %{exec} substitution,
  which means that the program may be run on a remote host depending on what
  %{exec} does.
  """
  if args is None:
    args = []
  with _makeConfigTest(config) as test:
    with open(test.getSourcePath(), 'w') as source:
      source.write(program)
    _, err, exitCode, _ = _executeScriptInternal(test, ['%{build}'])
    if exitCode != 0:
      raise ConfigurationCompilationError("Failed to build program, stderr is:\n{}".format(err))

    out, err, exitCode, _ = _executeScriptInternal(test, ["%{{run}} {}".format(' '.join(args))])
    if exitCode != 0:
      raise ConfigurationRuntimeError("Failed to run program, stderr is:\n{}".format(err))

    actualOut = re.search("# command output:\n(.+)\n$", out, flags=re.DOTALL)
    actualOut = actualOut.group(1) if actualOut else ""
    return actualOut

@_memoizeExpensiveOperation(lambda c, f: (c.substitutions, c.environment, f))
def hasCompileFlag(config, flag):
  """
  Return whether the compiler in the configuration supports a given compiler flag.

  This is done by executing the %{cxx} substitution with the given flag and
  checking whether that succeeds.
  """
  with _makeConfigTest(config) as test:
    out, err, exitCode, timeoutInfo = _executeScriptInternal(test, [
      "%{{cxx}} -xc++ {} -Werror -fsyntax-only %{{flags}} %{{compile_flags}} {}".format(os.devnull, flag)
    ])
    return exitCode == 0

@_memoizeExpensiveOperation(lambda c, s: (c.substitutions, c.environment, s))
def runScriptExitCode(config, script):
  """
  Runs the given script as a Lit test, and returns the exit code of the execution.

  The script must be a list of commands, each of which being something that
  could appear on the right-hand-side of a `RUN:` keyword.
  """
  with _makeConfigTest(config) as test:
    _, _, exitCode, _ = _executeScriptInternal(test, script)
    return exitCode

@_memoizeExpensiveOperation(lambda c, l: (c.substitutions, c.environment, l))
def hasAnyLocale(config, locales):
  """
  Return whether the runtime execution environment supports a given locale.
  Different systems may use different names for a locale, so this function checks
  whether any of the passed locale names is supported by setlocale() and returns
  true if one of them works.

  This is done by executing a program that tries to set the given locale using
  %{exec} -- this means that the command may be executed on a remote host
  depending on the %{exec} substitution.
  """
  program = """
    #include <stddef.h>
    #if defined(_LIBCPP_HAS_NO_LOCALIZATION)
      int main(int, char**) { return 1; }
    #else
      #include <locale.h>
      int main(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
          if (::setlocale(LC_ALL, argv[i]) != NULL) {
            return 0;
          }
        }
        return 1;
      }
    #endif
  """
  try:
    programOutput(config, program, args=[pipes.quote(l) for l in locales])
  except ConfigurationRuntimeError:
    return False
  return True

@_memoizeExpensiveOperation(lambda c, flags='': (c.substitutions, c.environment, flags))
def compilerMacros(config, flags=''):
  """
  Return a dictionary of predefined compiler macros.

  The keys are strings representing macros, and the values are strings
  representing what each macro is defined to.

  If the optional `flags` argument (a string) is provided, these flags will
  be added to the compiler invocation when generating the macros.
  """
  with _makeConfigTest(config) as test:
    with open(test.getSourcePath(), 'w') as sourceFile:
      # Make sure files like <__config> are included, since they can define
      # additional macros.
      sourceFile.write("#include <stddef.h>")
    unparsedOutput, err, exitCode, _ = _executeScriptInternal(test, [
      "%{{cxx}} %s -dM -E %{{flags}} %{{compile_flags}} {}".format(flags)
    ])
    if exitCode != 0:
      raise ConfigurationCompilationError("Failed to retrieve compiler macros, stderr is:\n{}".format(err))
    parsedMacros = dict()
    defines = (l.strip() for l in unparsedOutput.split('\n') if l.startswith('#define '))
    for line in defines:
      line = line[len('#define '):]
      macro, _, value = line.partition(' ')
      parsedMacros[macro] = value
    return parsedMacros

def featureTestMacros(config, flags=''):
  """
  Return a dictionary of feature test macros.

  The keys are strings representing feature test macros, and the values are
  integers representing the value of the macro.
  """
  allMacros = compilerMacros(config, flags)
  return {m: int(v.rstrip('LlUu')) for (m, v) in allMacros.items() if m.startswith('__cpp_')}

def _appendToSubstitution(substitutions, key, value):
  return [(k, v + ' ' + value) if k == key else (k, v) for (k, v) in substitutions]

def _prependToSubstitution(substitutions, key, value):
  return [(k, value + ' ' + v) if k == key else (k, v) for (k, v) in substitutions]


class ConfigAction(object):
  """
  This class represents an action that can be performed on a Lit TestingConfig
  object.

  Examples of such actions are adding or modifying substitutions, Lit features,
  etc. This class only provides the interface of such actions, and it is meant
  to be subclassed appropriately to create new actions.
  """
  def applyTo(self, config):
    """
    Applies the action to the given configuration.

    This should modify the configuration object in place, and return nothing.

    If applying the action to the configuration would yield an invalid
    configuration, and it is possible to diagnose it here, this method
    should produce an error. For example, it should be an error to modify
    a substitution in a way that we know for sure is invalid (e.g. adding
    a compiler flag when we know the compiler doesn't support it). Failure
    to do so early may lead to difficult-to-diagnose issues down the road.
    """
    pass

  def pretty(self, config, litParams):
    """
    Returns a short and human-readable string describing what this action does.

    This is used for logging purposes when running the test suite, so it should
    be kept concise.
    """
    pass


class AddFeature(ConfigAction):
  """
  This action defines the given Lit feature when running the test suite.

  The name of the feature can be a string or a callable, in which case it is
  called with the configuration to produce the feature name (as a string).
  """
  def __init__(self, name):
    self._name = name

  def _getName(self, config):
    name = self._name(config) if callable(self._name) else self._name
    if not isinstance(name, str):
      raise ValueError("Lit feature did not resolve to a string (got {})".format(name))
    return name

  def applyTo(self, config):
    config.available_features.add(self._getName(config))

  def pretty(self, config, litParams):
    return 'add Lit feature {}'.format(self._getName(config))


class AddFlag(ConfigAction):
  """
  This action adds the given flag to the %{flags} substitution.

  The flag can be a string or a callable, in which case it is called with the
  configuration to produce the actual flag (as a string).
  """
  def __init__(self, flag):
    self._getFlag = lambda config: flag(config) if callable(flag) else flag

  def applyTo(self, config):
    flag = self._getFlag(config)
    assert hasCompileFlag(config, flag), "Trying to enable flag {}, which is not supported".format(flag)
    config.substitutions = _appendToSubstitution(config.substitutions, '%{flags}', flag)

  def pretty(self, config, litParams):
    return 'add {} to %{{flags}}'.format(self._getFlag(config))


class AddFlagIfSupported(ConfigAction):
  """
  This action adds the given flag to the %{flags} substitution, only if
  the compiler supports the flag.

  The flag can be a string or a callable, in which case it is called with the
  configuration to produce the actual flag (as a string).
  """
  def __init__(self, flag):
    self._getFlag = lambda config: flag(config) if callable(flag) else flag

  def applyTo(self, config):
    flag = self._getFlag(config)
    if hasCompileFlag(config, flag):
      config.substitutions = _appendToSubstitution(config.substitutions, '%{flags}', flag)

  def pretty(self, config, litParams):
    return 'add {} to %{{flags}}'.format(self._getFlag(config))


class AddCompileFlag(ConfigAction):
  """
  This action adds the given flag to the %{compile_flags} substitution.

  The flag can be a string or a callable, in which case it is called with the
  configuration to produce the actual flag (as a string).
  """
  def __init__(self, flag):
    self._getFlag = lambda config: flag(config) if callable(flag) else flag

  def applyTo(self, config):
    flag = self._getFlag(config)
    assert hasCompileFlag(config, flag), "Trying to enable compile flag {}, which is not supported".format(flag)
    config.substitutions = _appendToSubstitution(config.substitutions, '%{compile_flags}', flag)

  def pretty(self, config, litParams):
    return 'add {} to %{{compile_flags}}'.format(self._getFlag(config))


class AddLinkFlag(ConfigAction):
  """
  This action appends the given flag to the %{link_flags} substitution.

  The flag can be a string or a callable, in which case it is called with the
  configuration to produce the actual flag (as a string).
  """
  def __init__(self, flag):
    self._getFlag = lambda config: flag(config) if callable(flag) else flag

  def applyTo(self, config):
    flag = self._getFlag(config)
    assert hasCompileFlag(config, flag), "Trying to enable link flag {}, which is not supported".format(flag)
    config.substitutions = _appendToSubstitution(config.substitutions, '%{link_flags}', flag)

  def pretty(self, config, litParams):
    return 'append {} to %{{link_flags}}'.format(self._getFlag(config))


class PrependLinkFlag(ConfigAction):
  """
  This action prepends the given flag to the %{link_flags} substitution.

  The flag can be a string or a callable, in which case it is called with the
  configuration to produce the actual flag (as a string).
  """
  def __init__(self, flag):
    self._getFlag = lambda config: flag(config) if callable(flag) else flag

  def applyTo(self, config):
    flag = self._getFlag(config)
    assert hasCompileFlag(config, flag), "Trying to enable link flag {}, which is not supported".format(flag)
    config.substitutions = _prependToSubstitution(config.substitutions, '%{link_flags}', flag)

  def pretty(self, config, litParams):
    return 'prepend {} to %{{link_flags}}'.format(self._getFlag(config))


class AddOptionalWarningFlag(ConfigAction):
  """
  This action adds the given warning flag to the %{compile_flags} substitution,
  if it is supported by the compiler.

  The flag can be a string or a callable, in which case it is called with the
  configuration to produce the actual flag (as a string).
  """
  def __init__(self, flag):
    self._getFlag = lambda config: flag(config) if callable(flag) else flag

  def applyTo(self, config):
    flag = self._getFlag(config)
    # Use -Werror to make sure we see an error about the flag being unsupported.
    if hasCompileFlag(config, '-Werror ' + flag):
      config.substitutions = _appendToSubstitution(config.substitutions, '%{compile_flags}', flag)

  def pretty(self, config, litParams):
    return 'add {} to %{{compile_flags}}'.format(self._getFlag(config))


class AddSubstitution(ConfigAction):
  """
  This action adds the given substitution to the Lit configuration.

  The substitution can be a string or a callable, in which case it is called
  with the configuration to produce the actual substitution (as a string).
  """
  def __init__(self, key, substitution):
    self._key = key
    self._getSub = lambda config: substitution(config) if callable(substitution) else substitution

  def applyTo(self, config):
    key = self._key
    sub = self._getSub(config)
    config.substitutions.append((key, sub))

  def pretty(self, config, litParams):
    return 'add substitution {} = {}'.format(self._key, self._getSub(config))


class Feature(object):
  """
  Represents a Lit available feature that is enabled whenever it is supported.

  A feature like this informs the test suite about a capability of the compiler,
  platform, etc. Unlike Parameters, it does not make sense to explicitly
  control whether a Feature is enabled -- it should be enabled whenever it
  is supported.
  """
  def __init__(self, name, actions=None, when=lambda _: True):
    """
    Create a Lit feature for consumption by a test suite.

    - name
        The name of the feature. This is what will end up in Lit's available
        features if the feature is enabled. This can be either a string or a
        callable, in which case it is passed the TestingConfig and should
        generate a string representing the name of the feature.

    - actions
        An optional list of ConfigActions to apply when the feature is supported.
        An AddFeature action is always created regardless of any actions supplied
        here -- these actions are meant to perform more than setting a corresponding
        Lit feature (e.g. adding compiler flags). If 'actions' is a callable, it
        is called with the current configuration object to generate the actual
        list of actions.

    - when
        A callable that gets passed a TestingConfig and should return a
        boolean representing whether the feature is supported in that
        configuration. For example, this can use `hasCompileFlag` to
        check whether the compiler supports the flag that the feature
        represents. If omitted, the feature will always be considered
        supported.
    """
    self._name = name
    self._actions = [] if actions is None else actions
    self._isSupported = when

  def _getName(self, config):
    name = self._name(config) if callable(self._name) else self._name
    if not isinstance(name, str):
      raise ValueError("Feature did not resolve to a name that's a string, got {}".format(name))
    return name

  def getActions(self, config):
    """
    Return the list of actions associated to this feature.

    If the feature is not supported, an empty list is returned.
    If the feature is supported, an `AddFeature` action is automatically added
    to the returned list of actions, in addition to any actions provided on
    construction.
    """
    if not self._isSupported(config):
      return []
    else:
      actions = self._actions(config) if callable(self._actions) else self._actions
      return [AddFeature(self._getName(config))] + actions

  def pretty(self, config):
    """
    Returns the Feature's name.
    """
    return self._getName(config)


def _str_to_bool(s):
  """
  Convert a string value to a boolean.

  True values are "y", "yes", "t", "true", "on" and "1", regardless of capitalization.
  False values are "n", "no", "f", "false", "off" and "0", regardless of capitalization.
  """
  trueVals = ["y", "yes", "t", "true", "on", "1"]
  falseVals = ["n", "no", "f", "false", "off", "0"]
  lower = s.lower()
  if lower in trueVals:
    return True
  elif lower in falseVals:
    return False
  else:
    raise ValueError("Got string '{}', which isn't a valid boolean".format(s))

def _parse_parameter(s, type):
  if type is bool and isinstance(s, str):
    return _str_to_bool(s)
  elif type is list and isinstance(s, str):
    return [x.strip() for x in s.split(',') if x.strip()]
  return type(s)


class Parameter(object):
  """
  Represents a parameter of a Lit test suite.

  Parameters are used to customize the behavior of test suites in a user
  controllable way. There are two ways of setting the value of a Parameter.
  The first one is to pass `--param <KEY>=<VALUE>` when running Lit (or
  equivalenlty to set `litConfig.params[KEY] = VALUE` somewhere in the
  Lit configuration files. This method will set the parameter globally for
  all test suites being run.

  The second method is to set `config.KEY = VALUE` somewhere in the Lit
  configuration files, which sets the parameter only for the test suite(s)
  that use that `config` object.

  Parameters can have multiple possible values, and they can have a default
  value when left unspecified. They can also have any number of ConfigActions
  associated to them, in which case the actions will be performed on the
  TestingConfig if the parameter is enabled. Depending on the actions
  associated to a Parameter, it may be an error to enable the Parameter
  if some actions are not supported in the given configuration. For example,
  trying to set the compilation standard to C++23 when `-std=c++23` is not
  supported by the compiler would be an error.
  """
  def __init__(self, name, type, help, actions, choices=None, default=None):
    """
    Create a Lit parameter to customize the behavior of a test suite.

    - name
        The name of the parameter that can be used to set it on the command-line.
        On the command-line, the parameter can be set using `--param <name>=<value>`
        when running Lit. This must be non-empty.

    - choices
        An optional non-empty set of possible values for this parameter. If provided,
        this must be anything that can be iterated. It is an error if the parameter
        is given a value that is not in that set, whether explicitly or through a
        default value.

    - type
        A callable that can be used to parse the value of the parameter given
        on the command-line. As a special case, using the type `bool` also
        allows parsing strings with boolean-like contents, and the type `list`
        will parse a string delimited by commas into a list of the substrings.

    - help
        A string explaining the parameter, for documentation purposes.
        TODO: We should be able to surface those from the Lit command-line.

    - actions
        A callable that gets passed the parsed value of the parameter (either
        the one passed on the command-line or the default one), and that returns
        a list of ConfigAction to perform given the value of the parameter.
        All the ConfigAction must be supported in the given configuration.

    - default
        An optional default value to use for the parameter when no value is
        provided on the command-line. If the default value is a callable, it
        is called with the TestingConfig and should return the default value
        for the parameter. Whether the default value is computed or specified
        directly, it must be in the 'choices' provided for that Parameter.
    """
    self._name = name
    if len(self._name) == 0:
      raise ValueError("Parameter name must not be the empty string")

    if choices is not None:
      self._choices = list(choices) # should be finite
      if len(self._choices) == 0:
        raise ValueError("Parameter '{}' must be given at least one possible value".format(self._name))
    else:
      self._choices = None

    self._parse = lambda x: _parse_parameter(x, type)
    self._help = help
    self._actions = actions
    self._default = default

  def _getValue(self, config, litParams):
    """
    Return the value of the parameter given the configuration objects.
    """
    param = getattr(config, self.name, None)
    param = litParams.get(self.name, param)
    if param is None and self._default is None:
      raise ValueError("Parameter {} doesn't have a default value, but it was not specified in the Lit parameters or in the Lit config".format(self.name))
    getDefault = lambda: self._default(config) if callable(self._default) else self._default

    if param is not None:
      (pretty, value) = (param, self._parse(param))
    else:
      value = getDefault()
      pretty = '{} (default)'.format(value)

    if self._choices and value not in self._choices:
      raise ValueError("Got value '{}' for parameter '{}', which is not in the provided set of possible choices: {}".format(value, self.name, self._choices))
    return (pretty, value)

  @property
  def name(self):
    """
    Return the name of the parameter.

    This is the name that can be used to set the parameter on the command-line
    when running Lit.
    """
    return self._name

  def getActions(self, config, litParams):
    """
    Return the list of actions associated to this value of the parameter.
    """
    (_, parameterValue) = self._getValue(config, litParams)
    return self._actions(parameterValue)

  def pretty(self, config, litParams):
    """
    Return a pretty representation of the parameter's name and value.
    """
    (prettyParameterValue, _) = self._getValue(config, litParams)
    return "{}={}".format(self.name, prettyParameterValue)
