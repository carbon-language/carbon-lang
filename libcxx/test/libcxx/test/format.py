import errno
import os
import tempfile
import time

import lit.formats  # pylint: disable=import-error


class LibcxxTestFormat(lit.formats.FileBasedTest):
    """
    Custom test format handler for use with the test format use by libc++.

    Tests fall into two categories:
      FOO.pass.cpp - Executable test which should compile, run, and exit with
                     code 0.
      FOO.fail.cpp - Negative test case which is expected to fail compilation.
    """

    def __init__(self, cxx_under_test, use_verify_for_fail,
                 cpp_flags, ld_flags, exec_env,
                 use_ccache=False):
        self.cxx_under_test = cxx_under_test
        self.use_verify_for_fail = use_verify_for_fail
        self.cpp_flags = list(cpp_flags)
        self.ld_flags = list(ld_flags)
        self.exec_env = dict(exec_env)
        self.use_ccache = use_ccache

    def execute(self, test, lit_config):
        while True:
            try:
                return self._execute(test, lit_config)
            except OSError, oe:
                if oe.errno != errno.ETXTBSY:
                    raise
                time.sleep(0.1)

    def _execute(self, test, lit_config):
        # Extract test metadata from the test file.
        requires = []
        unsupported = []
        use_verify = False
        with open(test.getSourcePath()) as f:
            for ln in f:
                if 'XFAIL:' in ln:
                    items = ln[ln.index('XFAIL:') + 6:].split(',')
                    test.xfails.extend([s.strip() for s in items])
                elif 'REQUIRES:' in ln:
                    items = ln[ln.index('REQUIRES:') + 9:].split(',')
                    requires.extend([s.strip() for s in items])
                elif 'UNSUPPORTED:' in ln:
                    items = ln[ln.index('UNSUPPORTED:') + 12:].split(',')
                    unsupported.extend([s.strip() for s in items])
                elif 'USE_VERIFY' in ln and self.use_verify_for_fail:
                    use_verify = True
                elif not ln.strip().startswith("//") and ln.strip():
                    # Stop at the first non-empty line that is not a C++
                    # comment.
                    break

        # Check that we have the required features.
        #
        # FIXME: For now, this is cribbed from lit.TestRunner, to avoid
        # introducing a dependency there. What we more ideally would like to do
        # is lift the "requires" handling to be a core lit framework feature.
        missing_required_features = [
            f for f in requires
            if f not in test.config.available_features
        ]
        if missing_required_features:
            return (lit.Test.UNSUPPORTED,
                    "Test requires the following features: %s" % (
                        ', '.join(missing_required_features),))

        unsupported_features = [f for f in unsupported
                                if f in test.config.available_features]
        if unsupported_features:
            return (lit.Test.UNSUPPORTED,
                    "Test is unsupported with the following features: %s" % (
                        ', '.join(unsupported_features),))

        # Evaluate the test.
        return self._evaluate_test(test, use_verify, lit_config)

    def _make_report(self, cmd, out, err, rc):  # pylint: disable=no-self-use
        report = "Command: %s\n" % cmd
        report += "Exit Code: %d\n" % rc
        if out:
            report += "Standard Output:\n--\n%s--\n" % out
        if err:
            report += "Standard Error:\n--\n%s--\n" % err
        report += '\n'
        return cmd, report, rc

    def _compile(self, output_path, source_path, use_verify=False):
        cmd = [self.cxx_under_test, '-c', '-o', output_path, source_path]
        cmd += self.cpp_flags
        if use_verify:
            cmd += ['-Xclang', '-verify']
        if self.use_ccache:
            cmd = ['ccache'] + cmd
        out, err, rc = lit.util.executeCommand(cmd)
        return cmd, out, err, rc

    def _link(self, exec_path, object_path):
        cmd = [self.cxx_under_test, '-o', exec_path, object_path]
        cmd += self.cpp_flags + self.ld_flags
        out, err, rc = lit.util.executeCommand(cmd)
        return cmd, out, err, rc

    def _compile_and_link(self, exec_path, source_path):
        object_file = tempfile.NamedTemporaryFile(suffix=".o", delete=False)
        object_path = object_file.name
        object_file.close()
        try:
            cmd, out, err, rc = self._compile(object_path, source_path)
            if rc != 0:
                return cmd, out, err, rc
            return self._link(exec_path, object_path)
        finally:
            try:
                os.remove(object_path)
            except OSError:
                pass

    def _build(self, exec_path, source_path, compile_only=False,
               use_verify=False):
        if compile_only:
            cmd, out, err, rc = self._compile(exec_path, source_path,
                                              use_verify)
        else:
            assert not use_verify
            cmd, out, err, rc = self._compile_and_link(exec_path, source_path)
        return self._make_report(cmd, out, err, rc)

    def _clean(self, exec_path):  # pylint: disable=no-self-use
        try:
            os.remove(exec_path)
        except OSError:
            pass

    def _run(self, exec_path, lit_config, in_dir=None):
        cmd = []
        if self.exec_env:
            cmd.append('env')
            cmd.extend('%s=%s' % (name, value)
                       for name, value in self.exec_env.items())
        cmd.append(exec_path)
        if lit_config.useValgrind:
            cmd = lit_config.valgrindArgs + cmd
        out, err, rc = lit.util.executeCommand(cmd, cwd=in_dir)
        return self._make_report(cmd, out, err, rc)

    def _evaluate_test(self, test, use_verify, lit_config):
        name = test.path_in_suite[-1]
        source_path = test.getSourcePath()
        source_dir = os.path.dirname(source_path)

        # Check what kind of test this is.
        assert name.endswith('.pass.cpp') or name.endswith('.fail.cpp')
        expected_compile_fail = name.endswith('.fail.cpp')

        # If this is a compile (failure) test, build it and check for failure.
        if expected_compile_fail:
            cmd, report, rc = self._build('/dev/null', source_path,
                                          compile_only=True,
                                          use_verify=use_verify)
            expected_rc = 0 if use_verify else 1
            if rc == expected_rc:
                return lit.Test.PASS, ""
            else:
                return (lit.Test.FAIL,
                        report + 'Expected compilation to fail!\n')
        else:
            exec_file = tempfile.NamedTemporaryFile(suffix="exe", delete=False)
            exec_path = exec_file.name
            exec_file.close()

            try:
                cmd, report, rc = self._build(exec_path, source_path)
                compile_cmd = cmd
                if rc != 0:
                    report += "Compilation failed unexpectedly!"
                    return lit.Test.FAIL, report

                cmd, report, rc = self._run(exec_path, lit_config,
                                            source_dir)
                if rc != 0:
                    report = "Compiled With: %s\n%s" % (compile_cmd, report)
                    report += "Compiled test failed unexpectedly!"
                    return lit.Test.FAIL, report
            finally:
                # Note that cleanup of exec_file happens in `_clean()`. If you
                # override this, cleanup is your reponsibility.
                self._clean(exec_path)
        return lit.Test.PASS, ""
