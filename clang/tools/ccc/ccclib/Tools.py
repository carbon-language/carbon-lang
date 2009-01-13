import os
import sys # FIXME: Shouldn't be needed.

import Arguments
import Jobs
import Types

class Tool(object):
    """Tool - A concrete implementation of an action."""

    eFlagsPipedInput = 1 << 0
    eFlagsPipedOutput = 1 << 1
    eFlagsIntegratedCPP = 1 << 2

    def __init__(self, name, flags = 0):
        self.name = name
        self.flags = flags

    def acceptsPipedInput(self):
        return not not (self.flags & Tool.eFlagsPipedInput)
    def canPipeOutput(self):
        return not not (self.flags & Tool.eFlagsPipedOutput)
    def hasIntegratedCPP(self):
        return not not (self.flags & Tool.eFlagsIntegratedCPP)

class GCC_Common_Tool(Tool):
    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist,
                     extraArgs):
        cmd_args = sum(map(arglist.render, args),[]) + extraArgs
        if arch:
            cmd_args.extend(arglist.render(arch))
        if isinstance(output, Jobs.PipedJob):
            cmd_args.extend(['-o', '-'])
        elif output is None:
            cmd_args.append('-fsyntax-only')
        else:
            cmd_args.extend(arglist.render(output))

        # Only pass -x if gcc will understand it; otherwise hope gcc
        # understands the suffix correctly. The main use case this
        # would go wrong in is for linker inputs if they happened to
        # have an odd suffix; really the only way to get this to
        # happen is a command like '-x foobar a.c' which will treat
        # a.c like a linker input.
        #
        # FIXME: For the linker case specifically, can we safely
        # convert inputs into '-Wl,' options?
        for input in inputs:
            if input.type.canBeUserSpecified:
                cmd_args.extend(['-x', input.type.name])

            if isinstance(input.source, Jobs.PipedJob):
                cmd_args.append('-')
            else:
                assert isinstance(input.source, Arguments.Arg)
                # If this is a linker input then assume we can forward
                # just by rendering.
                if input.source.opt.isLinkerInput:
                    cmd_args.extend(arglist.render(input.source))
                else:
                    cmd_args.extend(arglist.renderAsInput(input.source))

        jobs.addJob(Jobs.Command('gcc', cmd_args))

class GCC_PreprocessTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_PreprocessTool, self).__init__('gcc (cpp)',
                                                 (Tool.eFlagsPipedInput |
                                                  Tool.eFlagsPipedOutput))

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        return super(GCC_PreprocessTool, self).constructJob(phase, arch, jobs, inputs,
                                                            output, outputType, args, arglist,
                                                            ['-E'])

class GCC_CompileTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_CompileTool, self).__init__('gcc (cc1)',
                                              (Tool.eFlagsPipedInput |
                                               Tool.eFlagsPipedOutput |
                                               Tool.eFlagsIntegratedCPP))

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        return super(GCC_CompileTool, self).constructJob(phase, arch, jobs, inputs,
                                                         output, outputType, args, arglist,
                                                         ['-S'])

class GCC_PrecompileTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_PrecompileTool, self).__init__('gcc (pch)',
                                                 (Tool.eFlagsPipedInput |
                                                  Tool.eFlagsIntegratedCPP))

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        return super(GCC_PrecompileTool, self).constructJob(phase, arch, jobs, inputs,
                                                            output, outputType, args, arglist,
                                                            [])

class Darwin_AssembleTool(Tool):
    def __init__(self, toolChain):
        super(Darwin_AssembleTool, self).__init__('as',
                                                  Tool.eFlagsPipedInput)
        self.toolChain = toolChain

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        assert len(inputs) == 1
        assert outputType is Types.ObjectType

        input = inputs[0]

        cmd_args = []
        
        if (arglist.getLastArg(arglist.parser.gOption) or
            arglist.getLastArg(arglist.parser.g3Option)):
            cmd_args.append('--gstabs')

        # Derived from asm spec.
        if arch:
            cmd_args.extend(arglist.render(arch))
        cmd_args.append('-force_cpusubtype_ALL')
        if (arglist.getLastArg(arglist.parser.m_kernelOption) or
            arglist.getLastArg(arglist.parser.staticOption) or
            arglist.getLastArg(arglist.parser.f_appleKextOption)):
            if not arglist.getLastArg(arglist.parser.ZdynamicOption):
                cmd_args.append('-static')

        for arg in arglist.getArgs2(arglist.parser.WaOption,
                                    arglist.parser.XassemblerOption):
            cmd_args.extend(arglist.getValues(arg))

        cmd_args.extend(arglist.render(output))
        if isinstance(input.source, Jobs.PipedJob):
            cmd_args.append('-')
        else:
            cmd_args.extend(arglist.renderAsInput(input.source))
            
        # asm_final spec is empty.

        jobs.addJob(Jobs.Command(self.toolChain.getProgramPath('as'), 
                                 cmd_args))

class GCC_AssembleTool(GCC_Common_Tool):
    def __init__(self):
        # We can't generally assume the assembler can take or output
        # on pipes.
        super(GCC_AssembleTool, self).__init__('gcc (as)')

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        return super(GCC_AssembleTool, self).constructJob(phase, arch, jobs, inputs,
                                                          output, outputType, args, arglist,
                                                          ['-c'])

class GCC_LinkTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_LinkTool, self).__init__('gcc (ld)')

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        return super(GCC_LinkTool, self).constructJob(phase, arch, jobs, inputs,
                                                      output, outputType, args, arglist,
                                                      [])

class Darwin_X86_CompileTool(Tool):
    def __init__(self, toolChain):
        super(Darwin_X86_CompileTool, self).__init__('cc1',
                                                     (Tool.eFlagsPipedInput |
                                                      Tool.eFlagsPipedOutput |
                                                      Tool.eFlagsIntegratedCPP))
        self.toolChain = toolChain

    def addCPPArgs(self, cmd_args, arch, arglist):
        # Derived from cpp spec.

        if arglist.getLastArg(arglist.parser.staticOption):
            # The gcc spec is broken here, it refers to dynamic but
            # that has been translated. Start by being bug compatible.
            
            # if not arglist.getLastArg(arglist.parser.ZdynamicOption):
            cmd_args.append('-D__STATIC__')
        else:
            cmd_args.append('-D__DYNAMIC__')
        
        if arglist.getLastArg(arglist.parser.pthreadOption):
            cmd_args.append('-D_REENTRANT')
        
    def addCC1Args(self, cmd_args, arch, arglist):
        # Derived from cc1 spec.

        # FIXME: -fapple-kext seems to disable this too. Investigate.
        if (not arglist.getLastArg(arglist.parser.m_kernelOption) and
            not arglist.getLastArg(arglist.parser.staticOption) and
            not arglist.getLastArg(arglist.parser.m_dynamicNoPicOption)):
            cmd_args.append('-fPIC')

        # FIXME: Remove mthumb
        # FIXME: Remove mno-thumb

        # FIXME: As with ld, something else is going on. My best guess
        # is gcc is faking an -mmacosx-version-min
        # somewhere. Investigate.
        if (not arglist.getLastArg(arglist.parser.m_macosxVersionMinOption) and
            not arglist.getLastArg(arglist.parser.m_iphoneosVersionMinOption)):
            cmd_args.append('-mmacosx-version-min=' + 
                            self.toolChain.getMacosxVersionMin())

        # FIXME: Remove faltivec
        # FIXME: Remove mno-fused-madd
        # FIXME: Remove mlong-branch
        # FIXME: Remove mlongcall
        # FIXME: Remove mcpu=G4
        # FIXME: Remove mcpu=G5

        if (arglist.getLastArg(arglist.parser.gOption) and
            not arglist.getLastArg(arglist.parser.f_noEliminateUnusedDebugSymbolsOption)):
            cmd_args.append('-feliminate-unused-debug-symbols')

    def getBaseInputName(self, inputs, arglist):
        # FIXME: gcc uses a temporary name here when the base
        # input is stdin, but only in auxbase. Investigate.
        baseInputValue = arglist.getValue(inputs[0].baseInput)
        return os.path.basename(baseInputValue)
    
    def getBaseInputStem(self, inputs, arglist):
        return os.path.splitext(self.getBaseInputName(inputs, arglist))[0]

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, args, arglist):
        inputType = inputs[0].type
        assert not [i for i in inputs if i.type != inputType]

        usePP = False
        if inputType is Types.CType:
            cc1Name = 'cc1'
            usePP = True
        elif inputType is Types.CTypeNoPP:
            cc1Name = 'cc1'
            usePP = False
        elif inputType is Types.ObjCType:
            cc1Name = 'cc1obj'
            usePP = True
        elif inputType is Types.ObjCTypeNoPP:
            cc1Name = 'cc1obj'
            usePP = False
        elif inputType is Types.CXXType:
            cc1Name = 'cc1plus'
            usePP = True
        elif inputType is Types.CXXTypeNoPP:
            cc1Name = 'cc1plus'
            usePP = False
        elif inputType is Types.ObjCXXType:
            cc1Name = 'cc1objplus'
            usePP = True
        elif inputType is Types.ObjCXXTypeNoPP:
            cc1Name = 'cc1objplus'
            usePP = False
        else:
            raise RuntimeError,"Unexpected input type for Darwin compile tool."

        cmd_args = []
        if (arglist.getLastArg(arglist.parser.traditionalOption) or
            arglist.getLastArg(arglist.parser.f_traditionalOption)):
            raise ValueError,"-traditional is not supported without -E"

        if usePP:
            # Derived from cpp_unique_options.

            if (arglist.getLastArg(arglist.parser.COption) or
                arglist.getLastArg(arglist.parser.CCOption)):
                if not arglist.getLastArg(arglist.parser.EOption):
                    raise ValueError,"-C or -CC is not supported without -E"
            if not arglist.getLastArg(arglist.parser.QOption):
                cmd_args.append('-quiet')
            arglist.addAllArgs(cmd_args, arglist.parser.nostdincOption)
            arglist.addLastArg(cmd_args, arglist.parser.vOption)
            arglist.addAllArgs2(cmd_args, arglist.parser.IOption, arglist.parser.FOption)
            arglist.addLastArg(cmd_args, arglist.parser.POption)

            # FIXME: Handle %I properly.
            if arglist.getValue(arch) == 'x86_64':
                cmd_args.append('-imultilib')
                cmd_args.append('x86_64')

            if arglist.getLastArg(arglist.parser.MDOption):
                cmd_args.append('-MD')
                # FIXME: Think about this more.
                outputOpt = arglist.getLastArg(arglist.parser.oOption)
                if outputOpt:
                    base,ext = os.path.splitext(arglist.getValue(outputOpt))
                    cmd_args.append(base+'.d')
                else:
                    cmd_args.append(self.getBaseInputStem(inputs, arglist)+'.d')
            if arglist.getLastArg(arglist.parser.MMDOption):
                cmd_args.append('-MMD')
                # FIXME: Think about this more.
                outputOpt = arglist.getLastArg(arglist.parser.oOption)
                if outputOpt:
                    base,ext = os.path.splitext(arglist.getValue(outputOpt))
                    cmd_args.append(base+'.d')
                else:
                    cmd_args.append(self.getBaseInputStem(inputs, arglist)+'.d')
            arglist.addLastArg(cmd_args, arglist.parser.MOption)
            arglist.addLastArg(cmd_args, arglist.parser.MMOption)
            arglist.addAllArgs(cmd_args, arglist.parser.MFOption)
            arglist.addLastArg(cmd_args, arglist.parser.MGOption)
            arglist.addLastArg(cmd_args, arglist.parser.MPOption)
            arglist.addAllArgs(cmd_args, arglist.parser.MQOption)
            arglist.addAllArgs(cmd_args, arglist.parser.MTOption)
            if (not arglist.getLastArg(arglist.parser.MOption) and
                not arglist.getLastArg(arglist.parser.MMOption) and
                (arglist.getLastArg(arglist.parser.MDOption) or
                 arglist.getLastArg(arglist.parser.MMDOption))):
                outputOpt = arglist.getLastArg(arglist.parser.oOption)
                if outputOpt:
                    cmd_args.append('-MQ')
                    cmd_args.append(arglist.getValue(outputOpt))

            arglist.addLastArg(cmd_args, arglist.parser.remapOption)
            if arglist.getLastArg(arglist.parser.g3Option):
                cmd_args.append('-dD')
            arglist.addLastArg(cmd_args, arglist.parser.HOption)

            self.addCPPArgs(cmd_args, arch, arglist)

            arglist.addAllArgs3(cmd_args, 
                                arglist.parser.DOption,
                                arglist.parser.UOption,
                                arglist.parser.AOption)

            # FIXME: Add i*

            for input in inputs:
                if isinstance(input.source, Jobs.PipedJob):
                    cmd_args.append('-')
                else:
                    cmd_args.extend(arglist.renderAsInput(input.source))

            for arg in arglist.getArgs2(arglist.parser.WpOption,
                                        arglist.parser.XpreprocessorOption):
                cmd_args.extend(arglist.getValues(arg))

            if arglist.getLastArg(arglist.parser.f_mudflapOption):
                cmd_args.append('-D_MUDFLAP')
                cmd_args.append('-include')
                cmd_args.append('mf-runtime.h')

            if arglist.getLastArg(arglist.parser.f_mudflapthOption):
                cmd_args.append('-D_MUDFLAP')
                cmd_args.append('-D_MUDFLAPTH')
                cmd_args.append('-include')
                cmd_args.append('mf-runtime.h')
        else:
            cmd_args.append('-fpreprocessed')
            # FIXME: There is a spec command to remove
            # -fpredictive-compilation args here. Investigate.

            for input in inputs:
                if isinstance(input.source, Jobs.PipedJob):
                    cmd_args.append('-')
                else:
                    cmd_args.extend(arglist.renderAsInput(input.source))

        # Derived from cc1_options spec.
        if (arglist.getLastArg(arglist.parser.fastOption) or
            arglist.getLastArg(arglist.parser.fastfOption) or
            arglist.getLastArg(arglist.parser.fastcpOption)):
            cmd_args.append('-O3')
            
        if (arglist.getLastArg(arglist.parser.pgOption) and
            arglist.getLastArg(arglist.parser.f_omitFramePointerOption)):
            raise ValueError,"-pg and -fomit-frame-pointer are incompatible"

        self.addCC1Args(cmd_args, arch, arglist)

        if not arglist.getLastArg(arglist.parser.QOption):
            cmd_args.append('-quiet')

        cmd_args.append('-dumpbase')
        cmd_args.append(self.getBaseInputName(inputs, arglist))

        # FIXME: d*
        # FIXME: m*
        # FIXME: a*

        # FIXME: The goal is to use the user provided -o if that is
        # our final output, otherwise to drive from the original input
        # name.
        #
        # This implementation is close, but gcc also does this for -S
        # which is broken, and it would be nice to find a cleaner way
        # which doesn't introduce a dependency on the output argument
        # we are given.
        outputOpt = arglist.getLastArg(arglist.parser.oOption)
        if outputOpt and outputOpt is output:
            cmd_args.append('-auxbase-strip')
            cmd_args.append(arglist.getValue(outputOpt))
        else:
            cmd_args.append('-auxbase')
            cmd_args.append(self.getBaseInputStem(inputs, arglist))

        # FIXME: g*
            
        arglist.addAllArgs(cmd_args, arglist.parser.OOption)
        # FIXME: -Wall is getting some special treatment. Investigate.
        arglist.addAllArgs2(cmd_args, arglist.parser.WOption, arglist.parser.pedanticOption)
        arglist.addLastArg(cmd_args, arglist.parser.wOption)
        arglist.addAllArgs3(cmd_args, arglist.parser.stdOption, arglist.parser.ansiOption, arglist.parser.trigraphsOption)
        if arglist.getLastArg(arglist.parser.vOption):
            cmd_args.append('-version')
        if arglist.getLastArg(arglist.parser.pgOption):
            cmd_args.append('-p')
        arglist.addLastArg(cmd_args, arglist.parser.pOption)
        
        # FIXME: f*

        arglist.addLastArg(cmd_args, arglist.parser.undefOption)
        if arglist.getLastArg(arglist.parser.QnOption):
            cmd_args.append('-fno-ident')
         
        # FIXME: This isn't correct.
        #arglist.addLastArg(cmd_args, arglist.parser._helpOption)
        #arglist.addLastArg(cmd_args, arglist.parser._targetHelpOption)

        if isinstance(output, Jobs.PipedJob):
            cmd_args.extend(['-o', '-'])
        elif output is None:
            cmd_args.extend(['-o', '/dev/null'])
        else:
            cmd_args.extend(arglist.render(output))
  
        # FIXME: Still don't get what is happening here. Investigate.
        arglist.addAllArgs(cmd_args, arglist.parser._paramOption)

        if (arglist.getLastArg(arglist.parser.f_mudflapOption) or
            arglist.getLastArg(arglist.parser.f_mudflapthOption)):
            cmd_args.append('-fno-builtin')
            cmd_args.append('-fno-merge-constants')

        if arglist.getLastArg(arglist.parser.coverageOption):
            cmd_args.append('-fprofile-arcs')
            cmd_args.append('-ftest-coverage')

        jobs.addJob(Jobs.Command(self.toolChain.getProgramPath(cc1Name), 
                                 cmd_args))

        
class Darwin_X86_LinkTool(Tool):
    def __init__(self, toolChain):
        super(Darwin_X86_LinkTool, self).__init__('collect2')
        self.toolChain = toolChain

    def getMacosxVersionTuple(self, arglist):
        arg = arglist.getLastArg(arglist.parser.m_macosxVersionMinOption)
        if arg:
            version = arglist.getValue(arg)
            components = version.split('.')
            try:
                return tuple(map(int, components))
            except:
                raise ValueError,"invalid version number %r" % version
        else:
            major,minor,minorminor = self.toolChain.darwinVersion
            return (10, major-4, minor)

    def addDarwinArch(self, cmd_args, arch, arglist):
        # Derived from darwin_arch spec.
        cmd_args.append('-arch')
        # FIXME: The actual spec uses -m64 for this, but we want to
        # respect arch. Figure out what exactly gcc is doing.
        #if arglist.getLastArg(arglist.parser.m_64Option):
        if arglist.getValue(arch) == 'x86_64':
            cmd_args.append('x86_64')
        else:
            cmd_args.append('i386')

    def addDarwinSubArch(self, cmd_args, arch, arglist):
        # Derived from darwin_subarch spec, not sure what the
        # distinction exists for but at least for this chain it is the same.
        return self.addDarwinArch(cmd_args, arch, arglist)

    def addLinkArgs(self, cmd_args, arch, arglist):
        # Derived from link spec.
        if arglist.getLastArg(arglist.parser.staticOption):
            cmd_args.append('-static')
        else:
            cmd_args.append('-dynamic')
        if arglist.getLastArg(arglist.parser.f_gnuRuntimeOption):
            # FIXME: Replace -lobjc in forward args with
            # -lobjc-gnu. How do we wish to handle such things?
            pass

        if not arglist.getLastArg(arglist.parser.ZdynamiclibOption):
            if arglist.getLastArg(arglist.parser.Zforce_cpusubtype_ALLOption):
                self.addDarwinArch(cmd_args, arch, arglist)
                cmd_args.append('-force_cpusubtype_all')
            else:
                self.addDarwinSubArch(cmd_args, arch, arglist)
        
            if arglist.getLastArg(arglist.parser.ZbundleOption):
                cmd_args.append('-bundle')
            arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zbundle_loaderOption,
                                         '-bundle_loader')
            arglist.addAllArgs(cmd_args, arglist.parser.client_nameOption)
            if arglist.getLastArg(arglist.parser.compatibility_versionOption):
                # FIXME: Where should diagnostics go?
                print >>sys.stderr, "-compatibility_version only allowed with -dynamiclib"
                sys.exit(1)
            if arglist.getLastArg(arglist.parser.current_versionOption):
                print >>sys.stderr, "-current_version only allowed with -dynamiclib"
                sys.exit(1)
            if arglist.getLastArg(arglist.parser.Zforce_flat_namespaceOption):
                cmd_args.append('-force_flat_namespace')
            if arglist.getLastArg(arglist.parser.Zinstall_nameOption):
                print >>sys.stderr, "-install_name only allowed with -dynamiclib"
                sys.exit(1)
            arglist.addLastArg(cmd_args, arglist.parser.keep_private_externsOption)
            arglist.addLastArg(cmd_args, arglist.parser.private_bundleOption)
        else:
            cmd_args.append('-dylib')
            if arglist.getLastArg(arglist.parser.ZbundleOption):
                print >>sys.stderr, "-bundle not allowed with -dynamiclib"
                sys.exit(1)
            if arglist.getLastArg(arglist.parser.Zbundle_loaderOption):
                print >>sys.stderr, "-bundle_loader not allowed with -dynamiclib"
                sys.exit(1)
            if arglist.getLastArg(arglist.parser.client_nameOption):
                print >>sys.stderr, "-client_name not allowed with -dynamiclib"
                sys.exit(1)
            arglist.addAllArgsTranslated(cmd_args, arglist.parser.compatibility_versionOption,
                                         '-dylib_compatibility_version')
            arglist.addAllArgsTranslated(cmd_args, arglist.parser.current_versionOption,
                                         '-dylib_current_version')
 
            if arglist.getLastArg(arglist.parser.Zforce_cpusubtype_ALLOption):
                self.addDarwinArch(cmd_args, arch, arglist)
                # NOTE: We don't add -force_cpusubtype_ALL on this path. Ok.
            else:
                self.addDarwinSubArch(cmd_args, arch, arglist)
        
            if arglist.getLastArg(arglist.parser.Zforce_flat_namespaceOption):
                print >>sys.stderr, "-force_flat_namespace not allowed with -dynamiclib"
                sys.exit(1)

            arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zinstall_nameOption,
                                         '-dylib_install_name')

            if arglist.getLastArg(arglist.parser.keep_private_externsOption):
                print >>sys.stderr, "-keep_private_externs not allowed with -dynamiclib"
                sys.exit(1)
            if arglist.getLastArg(arglist.parser.private_bundleOption):
                print >>sys.stderr, "-private_bundle not allowed with -dynamiclib"
                sys.exit(1)

        if arglist.getLastArg(arglist.parser.Zall_loadOption):
            cmd_args.append('-all_load')

        arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zallowable_clientOption,
                                     '-allowable_client')

        if arglist.getLastArg(arglist.parser.Zbind_at_loadOption):
            cmd_args.append('-bind_at_load')

        if arglist.getLastArg(arglist.parser.Zdead_stripOption):
            cmd_args.append('-dead_strip')
        
        if arglist.getLastArg(arglist.parser.Zno_dead_strip_inits_and_termsOption):
            cmd_args.append('-no_dead_strip_inits_and_terms')
        
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zdylib_fileOption,
                                     '-dylib_file')

        if arglist.getLastArg(arglist.parser.ZdynamicOption):
            cmd_args.append('-dynamic')

        arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zexported_symbols_listOption,
                                     '-exported_symbols_list')

        if arglist.getLastArg(arglist.parser.Zflat_namespaceOption):
            cmd_args.append('-flat_namespace')

        arglist.addAllArgs(cmd_args, arglist.parser.headerpad_max_install_namesOption)
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zimage_baseOption,
                                     '-image_base')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.ZinitOption,
                                     '-init')

        if not arglist.getLastArg(arglist.parser.m_macosxVersionMinOption):
            if not arglist.getLastArg(arglist.parser.m_iphoneosVersionMinOption):
                # FIXME: I don't understand what is going on
                # here. This is supposed to come from
                # darwin_ld_minversion, but gcc doesn't seem to be
                # following that; it must be getting over-ridden
                # somewhere.
                cmd_args.append('-macosx_version_min')
                cmd_args.append(self.toolChain.getMacosxVersionMin())
        else:
            # addAll doesn't make sense here but this is what gcc
            # does.
            arglist.addAllArgsTranslated(cmd_args, arglist.parser.m_macosxVersionMinOption,
                                         '-macosx_version_min')

        arglist.addAllArgsTranslated(cmd_args, arglist.parser.m_iphoneosVersionMinOption,
                                     '-iphoneos_version_min')        
        arglist.addLastArg(cmd_args, arglist.parser.nomultidefsOption)
        
        if arglist.getLastArg(arglist.parser.Zmulti_moduleOption):
            cmd_args.append('-multi_module')
        
        if arglist.getLastArg(arglist.parser.Zsingle_moduleOption):
            cmd_args.append('-single_module')

        arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zmultiply_definedOption,
                                     '-multiply_defined')

        arglist.addAllArgsTranslated(cmd_args, arglist.parser.ZmultiplydefinedunusedOption,
                                     '-multiply_defined_unused')

        if arglist.getLastArg(arglist.parser.f_pieOption):
            cmd_args.append('-pie')

        arglist.addLastArg(cmd_args, arglist.parser.prebindOption)
        arglist.addLastArg(cmd_args, arglist.parser.noprebindOption)
        arglist.addLastArg(cmd_args, arglist.parser.nofixprebindingOption)
        arglist.addLastArg(cmd_args, arglist.parser.prebind_all_twolevel_modulesOption)
        arglist.addLastArg(cmd_args, arglist.parser.read_only_relocsOption)
        arglist.addAllArgs(cmd_args, arglist.parser.sectcreateOption)
        arglist.addAllArgs(cmd_args, arglist.parser.sectorderOption)
        arglist.addAllArgs(cmd_args, arglist.parser.seg1addrOption)
        arglist.addAllArgs(cmd_args, arglist.parser.segprotOption)
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.ZsegaddrOption,
                                     '-segaddr')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zsegs_read_only_addrOption,
                                     '-segs_read_only_addr')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zsegs_read_write_addrOption,
                                     '-segs_read_write_addr')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zseg_addr_tableOption,
                                     '-seg_addr_table')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zfn_seg_addr_table_filenameOption,
                                     '-fn_seg_addr_table_filename')
        arglist.addAllArgs(cmd_args, arglist.parser.sub_libraryOption)
        arglist.addAllArgs(cmd_args, arglist.parser.sub_umbrellaOption)
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.isysrootOption,
                                     '-syslibroot')
        arglist.addLastArg(cmd_args, arglist.parser.twolevel_namespaceOption)
        arglist.addLastArg(cmd_args, arglist.parser.twolevel_namespace_hintsOption)
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.ZumbrellaOption,
                                     '-umbrella')
        arglist.addAllArgs(cmd_args, arglist.parser.undefinedOption)
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zunexported_symbols_listOption,
                                     '-unexported_symbols_list')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.Zweak_reference_mismatchesOption,
                                     '-weak_reference_mismatches')
        
        if not arglist.getLastArg(arglist.parser.Zweak_reference_mismatchesOption):
            cmd_args.append('-weak_reference_mismatches')
            cmd_args.append('non-weak')

        arglist.addLastArg(cmd_args, arglist.parser.XOption)
        arglist.addAllArgs(cmd_args, arglist.parser.yOption)
        arglist.addLastArg(cmd_args, arglist.parser.wOption)
        arglist.addAllArgs(cmd_args, arglist.parser.pagezero_sizeOption)
        arglist.addAllArgs(cmd_args, arglist.parser.segs_read_Option)
        arglist.addLastArg(cmd_args, arglist.parser.seglinkeditOption)
        arglist.addLastArg(cmd_args, arglist.parser.noseglinkeditOption)
        arglist.addAllArgs(cmd_args, arglist.parser.sectalignOption)
        arglist.addAllArgs(cmd_args, arglist.parser.sectobjectsymbolsOption)
        arglist.addAllArgs(cmd_args, arglist.parser.segcreateOption)
        arglist.addLastArg(cmd_args, arglist.parser.whyloadOption)
        arglist.addLastArg(cmd_args, arglist.parser.whatsloadedOption)
        arglist.addAllArgs(cmd_args, arglist.parser.dylinker_install_nameOption)
        arglist.addLastArg(cmd_args, arglist.parser.dylinkerOption)
        arglist.addLastArg(cmd_args, arglist.parser.MachOption)

    def constructJob(self, phase, arch, jobs, inputs,
                     output, outputType, args, arglist):
        assert outputType is Types.ImageType

        # The logic here is derived from gcc's behavior; most of which
        # comes from specs (starting with link_command). Consult gcc
        # for more information.

        # FIXME: gcc's spec controls when this is done; certain things
        # like -filelist or -Wl, still trigger a link stage. I don't
        # quite understand how gcc decides to execute the linker,
        # investigate. Also, the spec references -fdump= which seems
        # to have disappeared?
        cmd_args = []

        # Not sure why this particular decomposition exists in gcc.
        self.addLinkArgs(cmd_args, arch, arglist)
        
        # This toolchain never accumlates options in specs, the only
        # place this gets used is to add -ObjC.
        if (arglist.getLastArg(arglist.parser.ObjCOption) or
            arglist.getLastArg(arglist.parser.f_objcOption)):
            cmd_args.append('-ObjC')
        if arglist.getLastArg(arglist.parser.ObjCXXOption):
            cmd_args.append('-ObjC')        

        # FIXME: gcc has %{x} in here. How could this ever happen?
        # Cruft?
        arglist.addLastArg(cmd_args, arglist.parser.dOption)
        arglist.addLastArg(cmd_args, arglist.parser.tOption)
        arglist.addLastArg(cmd_args, arglist.parser.ZOption)
        arglist.addLastArg(cmd_args, arglist.parser.uOption)
        arglist.addLastArg(cmd_args, arglist.parser.AOption)
        arglist.addLastArg(cmd_args, arglist.parser.eOption)
        arglist.addLastArg(cmd_args, arglist.parser.mOption)
        arglist.addLastArg(cmd_args, arglist.parser.rOption)

        cmd_args.extend(arglist.render(output))

        macosxVersion = self.getMacosxVersionTuple(arglist)
        if (not arglist.getLastArg(arglist.parser.AOption) and
            not arglist.getLastArg(arglist.parser.nostdlibOption) and
            not arglist.getLastArg(arglist.parser.nostartfilesOption)):
            # Derived from startfile spec.
            if arglist.getLastArg(arglist.parser.ZdynamiclibOption):
                # Derived from darwin_dylib1 spec.
                if arglist.getLastArg(arglist.parser.m_iphoneosVersionMinOption):
                    cmd_args.append('-ldylib1.o')
                else:
                    if macosxVersion < (10,5):
                        cmd_args.append('-ldylib1.o')
                    else:
                        cmd_args.append('-ldylib1.10.5.o')
            else:
                if arglist.getLastArg(arglist.parser.ZbundleOption):
                    if not arglist.getLastArg(arglist.parser.staticOption):
                        cmd_args.append('-lbundle1.o')
                else:
                    if arglist.getLastArg(arglist.parser.pgOption):
                        if arglist.getLastArg(arglist.parser.staticOption):
                            cmd_args.append('-lgcrt0.o')
                        else:
                            if arglist.getLastArg(arglist.parser.objectOption):
                                cmd_args.append('-lgcrt0.o')
                            else:
                                if arglist.getLastArg(arglist.parser.preloadOption):
                                    cmd_args.append('-lgcrt0.o')
                                else:
                                    cmd_args.append('-lgcrt1.o')

                                    # darwin_crt2 spec is empty.
                                    pass 
                    else:
                        if arglist.getLastArg(arglist.parser.staticOption):
                            cmd_args.append('-lcrt0.o')
                        else:
                            if arglist.getLastArg(arglist.parser.objectOption):
                                cmd_args.append('-lcrt0.o')
                            else:
                                if arglist.getLastArg(arglist.parser.preloadOption):
                                    cmd_args.append('-lcrt0.o')
                                else:
                                    # Derived from darwin_crt1 spec.
                                    if arglist.getLastArg(arglist.parser.m_iphoneosVersionMinOption):
                                        cmd_args.append('-lcrt1.o')
                                    else:
                                        if macosxVersion < (10,5):
                                            cmd_args.append('-lcrt1.o')
                                        else:
                                            cmd_args.append('-lcrt1.10.5.o')

                                    # darwin_crt2 spec is empty.
                                    pass 

            if arglist.getLastArg(arglist.parser.sharedLibgccOption):
                if not arglist.getLastArg(arglist.parser.m_iphoneosVersionMinOption):
                    if macosxVersion < (10,5):
                        # FIXME: gcc does a library search for this
                        # file, this will be be broken currently.
                        cmd_args.append('crt3.o')

        arglist.addAllArgs(cmd_args, arglist.parser.LOption)
        
        if arglist.getLastArg(arglist.parser.f_openmpOption):
            # This is more complicated in gcc...
            cmd_args.append('-lgomp')

        # FIXME: Derive these correctly.
        tcDir = self.toolChain.getToolChainDir()
        if arglist.getValue(arch) == 'x86_64':            
            cmd_args.extend(["-L/usr/lib/gcc/%s/x86_64" % tcDir,
                             "-L/usr/lib/gcc/%s/x86_64" % tcDir])
        cmd_args.extend(["-L/usr/lib/%s" % tcDir,
                         "-L/usr/lib/gcc/%s" % tcDir,
                         "-L/usr/lib/gcc/%s" % tcDir,
                         "-L/usr/lib/gcc/%s/../../../%s" % (tcDir,tcDir),
                         "-L/usr/lib/gcc/%s/../../.." % tcDir])

        for input in inputs:
            cmd_args.extend(arglist.renderAsInput(input.source))

        if (arglist.getLastArg(arglist.parser.f_profileArcsOption) or
            arglist.getLastArg(arglist.parser.f_profileGenerateOption) or
            arglist.getLastArg(arglist.parser.f_createProfileOption) or
            arglist.getLastArg(arglist.parser.coverageOption)):
            cmd_args.append('-lgcov')
        
        if arglist.getLastArg(arglist.parser.f_nestedFunctionsOption):
            cmd_args.append('-allow_stack_execute')

        if (not arglist.getLastArg(arglist.parser.nostdlibOption) and
            not arglist.getLastArg(arglist.parser.nodefaultlibsOption)):
            # link_ssp spec is empty.

            # Derived from libgcc spec.
            if arglist.getLastArg(arglist.parser.staticOption):
                cmd_args.append('-lgcc_static')
            elif arglist.getLastArg(arglist.parser.staticLibgccOption):
                cmd_args.append('-lgcc_eh')
                cmd_args.append('-lgcc')
            elif arglist.getLastArg(arglist.parser.m_iphoneosVersionMinOption):
                # Derived from darwin_iphoneos_libgcc spec.
                cmd_args.append('-lgcc_s.10.5')
                cmd_args.append('-lgcc')
            elif (arglist.getLastArg(arglist.parser.sharedLibgccOption) or
                  arglist.getLastArg(arglist.parser.f_exceptionsOption) or
                  arglist.getLastArg(arglist.parser.f_gnuRuntimeOption)):
                if macosxVersion < (10,5):
                    cmd_args.append('-lgcc_s.10.4')
                else:
                    cmd_args.append('-lgcc_s.10.5')
                cmd_args.append('-lgcc')
            else:
                if macosxVersion < (10,5) and macosxVersion >= (10,3,9):
                    cmd_args.append('-lgcc_s.10.4')
                if macosxVersion >= (10,5):
                    cmd_args.append('-lgcc_s.10.5')
                cmd_args.append('-lgcc')

            # Derived from lib spec.
            if not arglist.getLastArg(arglist.parser.staticOption):
                cmd_args.append('-lSystem')

        if (not arglist.getLastArg(arglist.parser.AOption) and
            not arglist.getLastArg(arglist.parser.nostdlibOption) and
            not arglist.getLastArg(arglist.parser.nostartfilesOption)):
            # endfile_spec is empty.
            pass

        arglist.addAllArgs(cmd_args, arglist.parser.TOption)
        arglist.addAllArgs(cmd_args, arglist.parser.FOption)

        jobs.addJob(Jobs.Command(self.toolChain.getProgramPath('collect2'), 
                                 cmd_args))

        # FIXME: We need to add a dsymutil job here in some particular
        # cases (basically whenever we have a c-family input we are
        # compiling, I think). Find out why this is the condition, and
        # implement. See link_command spec for more details.

class LipoTool(Tool):
    def __init__(self):
        super(LipoTool, self).__init__('lipo')

    def constructJob(self, phase, arch, jobs, inputs,
                     output, outputType, args, arglist):
        assert outputType is Types.ImageType

        cmd_args = ['-create']
        cmd_args.extend(arglist.render(output))
        for input in inputs:
            cmd_args.extend(arglist.renderAsInput(input.source))
        jobs.addJob(Jobs.Command('lipo', cmd_args))
