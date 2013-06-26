#!/usr/bin/python

def c_compiler_rule(b, name, description, compiler, flags):
  command = "%s -MMD -MF $out.d %s -c -o $out $in" % (compiler, flags)
  b.rule(name, command, description + " $out", depfile="$out.d")

from optparse import OptionParser
import os
from subprocess import *
import sys

srcdir = os.path.dirname(sys.argv[0])

sys.path.insert(0, os.path.join(srcdir, 'build'))
import metabuild

p = OptionParser()
p.add_option('--with-llvm-config', metavar='PATH',
             help='use given llvm-config script')
p.add_option('--prefix', metavar='PATH',
             help='install to given prefix')
p.add_option('-g', metavar='GENERATOR', default='make',
             help='use given generator (default: make)')
(options, args) = p.parse_args()

llvm_config_exe = options.with_llvm_config or "llvm-config"

def llvm_config(args):
  try:
    proc = Popen([llvm_config_exe] + args, stdout=PIPE)
    return proc.communicate()[0].rstrip().replace('\n', ' ')
  except OSError:
    print "Error executing llvm-config."
    print "Please ensure that llvm-config is in your $PATH, or use --with-llvm-config."
    sys.exit(1)

llvm_bindir = llvm_config(['--bindir'])
llvm_core_libs = llvm_config(['--libs', 'core', 'bitreader', 'bitwriter']) + ' ' + \
                 llvm_config(['--ldflags'])
llvm_cxxflags = llvm_config(['--cxxflags']) + ' -fno-exceptions -fno-rtti'

llvm_clang = os.path.join(llvm_bindir, 'clang')
llvm_link = os.path.join(llvm_bindir, 'llvm-link')
llvm_opt = os.path.join(llvm_bindir, 'opt')

default_targets = ['nvptx--nvidiacl', 'nvptx64--nvidiacl', 'r600--']

targets = args
if not targets:
  targets = default_targets

b = metabuild.from_name(options.g)

b.rule("LLVM_AS", "%s -o $out $in" % os.path.join(llvm_bindir, "llvm-as"),
       'LLVM-AS $out')
b.rule("LLVM_LINK", command = llvm_link + " -o $out $in",
       description = 'LLVM-LINK $out')
b.rule("OPT", command = llvm_opt + " -O3 -o $out $in",
       description = 'OPT $out')

c_compiler_rule(b, "LLVM_TOOL_CXX", 'CXX', 'c++', llvm_cxxflags)
b.rule("LLVM_TOOL_LINK", "c++ -o $out $in %s" % llvm_core_libs, 'LINK $out')

prepare_builtins = os.path.join('utils', 'prepare-builtins')
b.build(os.path.join('utils', 'prepare-builtins.o'), "LLVM_TOOL_CXX",
        os.path.join(srcdir, 'utils', 'prepare-builtins.cpp'))
b.build(prepare_builtins, "LLVM_TOOL_LINK",
        os.path.join('utils', 'prepare-builtins.o'))

b.rule("PREPARE_BUILTINS", "%s -o $out $in" % prepare_builtins,
       'PREPARE-BUILTINS $out')

manifest_deps = set([sys.argv[0], os.path.join(srcdir, 'build', 'metabuild.py'),
                     os.path.join(srcdir, 'build', 'ninja_syntax.py')])

install_files = []
install_deps = []

for target in targets:
  (t_arch, t_vendor, t_os) = target.split('-')
  archs = [t_arch]
  if t_arch == 'nvptx' or t_arch == 'nvptx64':
    archs.append('ptx')
  archs.append('generic')

  subdirs = []
  for arch in archs:
    subdirs.append("%s-%s-%s" % (arch, t_vendor, t_os))
    subdirs.append("%s-%s" % (arch, t_os))
    subdirs.append(arch)

  incdirs = filter(os.path.isdir,
               [os.path.join(srcdir, subdir, 'include') for subdir in subdirs])
  libdirs = filter(lambda d: os.path.isfile(os.path.join(d, 'SOURCES')),
                   [os.path.join(srcdir, subdir, 'lib') for subdir in subdirs])

  clang_cl_includes = ' '.join(["-I%s" % incdir for incdir in incdirs])
  install_files += [(incdir, incdir[len(srcdir)+1:]) for incdir in incdirs]

  # The rule for building a .bc file for the specified architecture using clang.
  clang_bc_flags = "-target %s -I`dirname $in` %s " \
                   "-Dcl_clang_storage_class_specifiers " \
                   "-Dcl_khr_fp64 " \
                   "-emit-llvm" % (target, clang_cl_includes)
  clang_bc_rule = "CLANG_CL_BC_" + target
  c_compiler_rule(b, clang_bc_rule, "LLVM-CC", llvm_clang, clang_bc_flags)
  
  objects = []
  sources_seen = set()

  for libdir in libdirs:
    subdir_list_file = os.path.join(libdir, 'SOURCES')
    manifest_deps.add(subdir_list_file)
    for src in open(subdir_list_file).readlines():
      src = src.rstrip()
      if src not in sources_seen:
        sources_seen.add(src)
        obj = os.path.join(target, 'lib', src + '.bc')
        objects.append(obj)
        src_file = os.path.join(libdir, src)
        ext = os.path.splitext(src)[1]
        if ext == '.ll':
          b.build(obj, 'LLVM_AS', src_file)
        else:
          b.build(obj, clang_bc_rule, src_file)

  builtins_link_bc = os.path.join(target, 'lib', 'builtins.link.bc')
  builtins_opt_bc = os.path.join(target, 'lib', 'builtins.opt.bc')
  builtins_bc = os.path.join(target, 'lib', 'builtins.bc')
  b.build(builtins_link_bc, "LLVM_LINK", objects)
  b.build(builtins_opt_bc, "OPT", builtins_link_bc)
  b.build(builtins_bc, "PREPARE_BUILTINS", builtins_opt_bc, prepare_builtins)
  install_files.append((builtins_bc, builtins_bc))
  install_deps.append(builtins_bc)
  b.default(builtins_bc)

if options.prefix:
  install_cmd = ' && '.join(['mkdir -p %(dst)s && cp -r %(src)s %(dst)s' % 
                             {'src': file,
                              'dst': os.path.join(options.prefix,
                                                  os.path.dirname(dest))}
                             for (file, dest) in install_files])
  b.rule('install', command = install_cmd, description = 'INSTALL')
  b.build('install', 'install', install_deps)

b.rule("configure", command = ' '.join(sys.argv), description = 'CONFIGURE',
       generator = True)
b.build(b.output_filename(), 'configure', list(manifest_deps))

b.finish()
