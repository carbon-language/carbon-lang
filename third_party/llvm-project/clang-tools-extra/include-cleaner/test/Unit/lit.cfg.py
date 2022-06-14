import lit.formats
config.name = "clangIncludeCleaner Unit Tests"
config.test_format = lit.formats.GoogleTest('.', 'Tests')
config.test_source_root = config.clang_include_cleaner_binary_dir + "/unittests"
config.test_exec_root = config.clang_include_cleaner_binary_dir + "/unittests"

# Point the dynamic loader at dynamic libraries in 'lib'.
# FIXME: it seems every project has a copy of this logic. Move it somewhere.
import platform
if platform.system() == 'Darwin':
    shlibpath_var = 'DYLD_LIBRARY_PATH'
elif platform.system() == 'Windows':
    shlibpath_var = 'PATH'
else:
    shlibpath_var = 'LD_LIBRARY_PATH'
config.environment[shlibpath_var] = os.path.pathsep.join((
    "@SHLIBDIR@", "@LLVM_LIBS_DIR@",
    config.environment.get(shlibpath_var,'')))
