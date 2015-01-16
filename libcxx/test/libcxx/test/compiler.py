
import lit.util


class CXXCompiler(object):
    def __init__(self, path, flags=[], compile_flags=[], link_flags=[], use_ccache=False):
        self.path = path
        self.flags = list(flags)
        self.compile_flags = list(compile_flags)
        self.link_flags = list(link_flags)
        self.use_ccache = use_ccache
        self.type = None
        self.version = (None, None, None)
        self._initTypeAndVersion()

    def _initTypeAndVersion(self):
        # Get compiler type and version
        macros = self.dumpMacros()
        if macros is None:
            return
        compiler_type = None
        major_ver = minor_ver = patchlevel = None
        if '__clang__' in macros.keys():
            compiler_type = 'clang'
            # Treat apple's llvm fork differently.
            if '__apple_build_version__' in macros.keys():
                compiler_type = 'apple-clang'
            major_ver = macros['__clang_major__']
            minor_ver = macros['__clang_minor__']
            patchlevel = macros['__clang_patchlevel__']
        elif '__GNUC__' in macros.keys():
            compiler_type = 'gcc'
            major_ver = macros['__GNUC__']
            minor_ver = macros['__GNUC_MINOR__']
            patchlevel = macros['__GNUC_PATCHLEVEL__']
        self.type = compiler_type
        self.version = (major_ver, minor_ver, patchlevel)

    def _basicCmd(self, infiles, out, is_link=False):
        cmd = []
        if self.use_ccache and not is_link:
            cmd += ['ccache']
        cmd += [self.path]
        if out is not None:
            cmd += ['-o', out]
        if isinstance(infiles, list):
            cmd += infiles
        elif isinstance(infiles, str):
            cmd += [infiles]
        else:
            raise TypeError('infiles must be a string or list')
        return cmd

    def preprocessCmd(self, infiles, out=None, flags=[]):
        cmd = self._basicCmd(infiles, out) + ['-x', 'c++', '-E']
        cmd += self.flags + self.compile_flags + flags
        return cmd

    def compileCmd(self, infiles, out=None, flags=[]):
        cmd = self._basicCmd(infiles, out) + ['-x', 'c++', '-c']
        cmd += self.flags + self.compile_flags + flags
        return cmd

    def linkCmd(self, infiles, out=None, flags=[]):
        cmd = self._basicCmd(infiles, out, is_link=True)
        cmd += self.flags + self.link_flags + flags
        return cmd

    def compileLinkCmd(self, infiles, out=None, flags=[]):
        cmd = self._basicCmd(infiles, out, is_link=True) + ['-x', 'c++']
        cmd += self.flags + self.compile_flags + self.link_flags + flags
        return cmd

    def preprocess(self, infiles, out=None, flags=[], env=None, cwd=None):
        cmd = self.preprocessCmd(infiles, out, flags)
        out, err, rc = lit.util.executeCommand(cmd, env=env, cwd=cwd)
        return cmd, out, err, rc

    def compile(self, infiles, out=None, flags=[], env=None, cwd=None):
        cmd = self.compileCmd(infiles, out, flags)
        out, err, rc = lit.util.executeCommand(cmd, env=env, cwd=cwd)
        return cmd, out, err, rc

    def link(self, infiles, out=None, flags=[], env=None, cwd=None):
        cmd = self.linkCmd(infiles, out, flags)
        out, err, rc = lit.util.executeCommand(cmd, env=env, cwd=cwd)
        return cmd, out, err, rc

    def compileLink(self, infiles, out=None, flags=[], env=None, cwd=None):
        cmd = self.compileLinkCmd(infiles, out, flags)
        out, err, rc = lit.util.executeCommand(cmd, env=env, cwd=cwd)
        return cmd, out, err, rc

    def dumpMacros(self, infiles=None, flags=[], env=None, cwd=None):
        if infiles is None:
            infiles = '/dev/null'
        flags = ['-dM'] + flags
        cmd, out, err, rc = self.preprocess(infiles, flags=flags, env=env,
                                            cwd=cwd)
        if rc != 0:
            return None
        parsed_macros = dict()
        lines = [l.strip() for l in out.split('\n') if l.strip()]
        for l in lines:
            assert l.startswith('#define ')
            l = l[len('#define '):]
            macro, _, value = l.partition(' ')
            parsed_macros[macro] = value
        return parsed_macros

    def getTriple(self):
        cmd = [self.path] + self.flags + ['-dumpmachine']
        return lit.util.capture(cmd).strip()
