import os
import sys # FIXME: Shouldn't be needed.

import Arguments
import Jobs
import Phases
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
    def getGCCExtraArgs(self):
        return []

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, arglist, linkingOutput):
        cmd_args = []
        for arg in arglist.args:
            if arg.opt.forwardToGCC():
                cmd_args.extend(arglist.render(arg))

        cmd_args.extend(self.getGCCExtraArgs())
        if arch:
            cmd_args.extend(arglist.render(arch))
        if isinstance(output, Jobs.PipedJob):
            cmd_args.extend(['-o', '-'])
        elif isinstance(phase.phase, Phases.SyntaxOnlyPhase):
            cmd_args.append('-fsyntax-only')
        else:
            assert output
            cmd_args.extend(arglist.render(output))

        if (isinstance(self, GCC_LinkTool) and
            linkingOutput):
            cmd_args.append('-Wl,-arch_multiple')
            cmd_args.append('-Wl,-final_output,' + 
                            arglist.getValue(linkingOutput))

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

    def getGCCExtraArgs(self):
        return ['-E']

class GCC_CompileTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_CompileTool, self).__init__('gcc (cc1)',
                                              (Tool.eFlagsPipedInput |
                                               Tool.eFlagsPipedOutput |
                                               Tool.eFlagsIntegratedCPP))

    def getGCCExtraArgs(self):
        return ['-S']

class GCC_PrecompileTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_PrecompileTool, self).__init__('gcc (pch)',
                                                 (Tool.eFlagsPipedInput |
                                                  Tool.eFlagsIntegratedCPP))

    def getGCCExtraArgs(self):
        return []

class GCC_AssembleTool(GCC_Common_Tool):
    def __init__(self):
        # We can't generally assume the assembler can take or output
        # on pipes.
        super(GCC_AssembleTool, self).__init__('gcc (as)')

    def getGCCExtraArgs(self):
        return ['-c']

class GCC_LinkTool(GCC_Common_Tool):
    def __init__(self):
        super(GCC_LinkTool, self).__init__('gcc (ld)')

class Darwin_AssembleTool(Tool):
    def __init__(self, toolChain):
        super(Darwin_AssembleTool, self).__init__('as',
                                                  Tool.eFlagsPipedInput)
        self.toolChain = toolChain

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, arglist, linkingOutput):
        assert len(inputs) == 1
        assert outputType is Types.ObjectType

        input = inputs[0]

        cmd_args = []
        
        # Bit of a hack, this is only used for original inputs.
        if input.isOriginalInput():
            if arglist.getLastArg(arglist.parser.gGroup):
                cmd_args.append('--gstabs')

        # Derived from asm spec.
        if arch:
            cmd_args.extend(arglist.render(arch))
        cmd_args.append('-force_cpusubtype_ALL')
        if (arglist.getLastArg(arglist.parser.m_kernelOption) or
            arglist.getLastArg(arglist.parser.staticOption) or
            arglist.getLastArg(arglist.parser.f_appleKextOption)):
            if not arglist.getLastArg(arglist.parser.dynamicOption):
                cmd_args.append('-static')

        for arg in arglist.getArgs2(arglist.parser.WaOption,
                                    arglist.parser.XassemblerOption):
            cmd_args.extend(arglist.getValues(arg))

        cmd_args.extend(arglist.render(output))
        if isinstance(input.source, Jobs.PipedJob):
            pass
        else:
            cmd_args.extend(arglist.renderAsInput(input.source))
            
        # asm_final spec is empty.

        jobs.addJob(Jobs.Command(self.toolChain.getProgramPath('as'), 
                                 cmd_args))

class Clang_CompileTool(Tool):
    def __init__(self, toolChain):
        super(Clang_CompileTool, self).__init__('clang',
                                   (Tool.eFlagsPipedInput |
                                    Tool.eFlagsPipedOutput |
                                    Tool.eFlagsIntegratedCPP))
        self.toolChain = toolChain

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, arglist, linkingOutput):
        cmd_args = []

        patchOutputNameForPTH = False

        if isinstance(phase.phase, Phases.AnalyzePhase):
            cmd_args.append('-analyze')
        elif isinstance(phase.phase, Phases.SyntaxOnlyPhase):
            cmd_args.append('-fsyntax-only')
        elif outputType is Types.AsmTypeNoPP:
            cmd_args.append('-S')
        elif outputType is Types.PCHType:
            # No special option needed, driven by -x. However, we
            # patch the output name to try and not conflict with gcc.
            patchOutputNameForPTH = True

            # FIXME: This is a total hack. Copy the input header file
            # to the output, so that it can be -include'd by clang.
            assert len(inputs) == 1
            assert not isinstance(output, Jobs.PipedJob)
            assert not isinstance(inputs[0].source, Jobs.PipedJob)
            inputPath = arglist.getValue(inputs[0].source)
            outputPath = os.path.join(os.path.dirname(arglist.getValue(output)),
                                      os.path.basename(inputPath))
            # Only do copy when the output doesn't exist.
            if not os.path.exists(outputPath):
                import shutil
                shutil.copyfile(inputPath, outputPath)
        else:
            raise ValueError,"Unexpected output type for clang tool."

        if isinstance(phase.phase, Phases.AnalyzePhase):
            # Add default argument set.
            #
            # FIXME: Move into clang?
            cmd_args.extend(['-warn-dead-stores',
                             '-checker-cfref',
                             '-warn-objc-methodsigs',
                             '-warn-objc-missing-dealloc',
                             '-warn-objc-unused-ivars'])
            
            cmd_args.append('-analyzer-output-plist')

            # Add -WA, arguments when running as analyzer.
            for arg in arglist.getArgs(arglist.parser.WAOption):
                cmd_args.extend(arglist.renderAsInput(arg))
        else:
            # Perform argument translation for LLVM backend. This
            # performs some care in reconciling with llvm-gcc. The
            # issue is that llvm-gcc translates these options based on
            # the values in cc1, whereas we are processing based on
            # the driver arguments.
            #
            # FIXME: This is currently broken for -f flags when -fno
            # variants are present.

            # This comes from the default translation the driver + cc1
            # would do to enable flag_pic.
            # 
            # FIXME: Centralize this code.
            picEnabled = (arglist.getLastArg(arglist.parser.f_PICOption) or
                          arglist.getLastArg(arglist.parser.f_picOption) or
                          arglist.getLastArg(arglist.parser.f_PIEOption) or
                          arglist.getLastArg(arglist.parser.f_pieOption) or
                          (not arglist.getLastArg(arglist.parser.m_kernelOption) and
                          not arglist.getLastArg(arglist.parser.staticOption) and
                          not arglist.getLastArg(arglist.parser.m_dynamicNoPicOption)))

            archName = arglist.getValue(arch)
            if (archName == 'x86_64' or 
                picEnabled):
                cmd_args.append('--relocation-model=pic')
            else:
                cmd_args.append('--relocation-model=static')

            if arglist.getLastArg(arglist.parser.f_timeReportOption):
                cmd_args.append('--time-passes')
            # FIXME: Set --enable-unsafe-fp-math.
            if not arglist.getLastArg(arglist.parser.f_omitFramePointerOption):
                cmd_args.append('--disable-fp-elim')
            if not arglist.getLastArg(arglist.parser.f_zeroInitializedInBssOption):
                cmd_args.append('--nozero-initialized-in-bss')
            if arglist.getLastArg(arglist.parser.dAOption):
                cmd_args.append('--asm-verbose')
            if arglist.getLastArg(arglist.parser.f_debugPassStructureOption):
                cmd_args.append('--debug-pass=Structure')
            if arglist.getLastArg(arglist.parser.f_debugPassArgumentsOption):
                cmd_args.append('--debug-pass=Arguments')
            # FIXME: set --inline-threshhold=50 if (optimize_size || optimize < 3)
            if arglist.getLastArg(arglist.parser.f_unwindTablesOption):
                cmd_args.append('--unwind-tables')

            arg = arglist.getLastArg(arglist.parser.f_limitedPrecisionOption)
            if arg:
                cmd_args.append('--limit-float-precision')
                cmd_args.append(arglist.getValue(arg))
            
            # FIXME: Add --stack-protector-buffer-size=<xxx> on -fstack-protect.

        arglist.addAllArgs(cmd_args, arglist.parser.vOption)
        arglist.addAllArgs2(cmd_args, arglist.parser.DOption, arglist.parser.UOption)
        arglist.addAllArgs2(cmd_args, arglist.parser.IOption, arglist.parser.FOption)
        arglist.addAllArgs(cmd_args, arglist.parser.m_macosxVersionMinOption)

        # Special case debug options to only pass -g to clang. This is
        # wrong.
        if arglist.getLastArg(arglist.parser.gGroup):
            cmd_args.append('-g')

        arglist.addLastArg(cmd_args, arglist.parser.nostdincOption)

        # FIXME: Clang isn't going to accept just anything here.
        arglist.addAllArgs(cmd_args, arglist.parser.iGroup)

        # Automatically load .pth files which match -include options.
        for arg in arglist.getArgs(arglist.parser.includeOption):
            pthPath = arglist.getValue(arg) + '.pth'
            if os.path.exists(pthPath):
                cmd_args.append('-token-cache')
                cmd_args.append(pthPath)

        # FIXME: Dehardcode this.
        cmd_args.append('-fblocks')

        arglist.addAllArgs(cmd_args, arglist.parser.OOption)
        arglist.addAllArgs2(cmd_args, arglist.parser.ClangWGroup, arglist.parser.pedanticGroup)
        arglist.addLastArg(cmd_args, arglist.parser.wOption)
        arglist.addAllArgs3(cmd_args, arglist.parser.stdOption, arglist.parser.ansiOption, arglist.parser.trigraphsOption)

        arglist.addAllArgs(cmd_args, arglist.parser.f_objcGcOption)
        arglist.addAllArgs(cmd_args, arglist.parser.f_objcGcOnlyOption)
        arglist.addAllArgs(cmd_args, arglist.parser.f_nextRuntimeOption)
        arglist.addAllArgs(cmd_args, arglist.parser.f_gnuRuntimeOption)
        arglist.addLastArg(cmd_args, arglist.parser.f_exceptionsOption)
        arglist.addLastArg(cmd_args, arglist.parser.f_laxVectorConversionsOption)
        arglist.addLastArg(cmd_args, arglist.parser.f_msExtensionsOption)
        arglist.addLastArg(cmd_args, arglist.parser.f_noCaretDiagnosticsOption)
        arglist.addLastArg(cmd_args, arglist.parser.f_noShowColumnOption)
        arglist.addLastArg(cmd_args, arglist.parser.f_pascalStringsOption)
        arglist.addLastArg(cmd_args, arglist.parser.f_writableStringsOption)

        if arch is not None:
            cmd_args.extend(arglist.render(arch))

        if isinstance(output, Jobs.PipedJob):
            cmd_args.extend(['-o', '-'])
        else:
            if patchOutputNameForPTH:
                base,suffix = os.path.splitext(arglist.getValue(output))
                if suffix == '.gch':
                    suffix = '.pth'
                cmd_args.append('-o')
                cmd_args.append(base + suffix)
            elif output:
                cmd_args.extend(arglist.render(output))

        for input in inputs:
            cmd_args.append('-x')
            cmd_args.append(input.type.name)
            if isinstance(input.source, Jobs.PipedJob):
                cmd_args.append('-')
            else:
                cmd_args.extend(arglist.renderAsInput(input.source))
            
        jobs.addJob(Jobs.Command(self.toolChain.getProgramPath('clang'), 
                                 cmd_args))
        
class Darwin_X86_CC1Tool(Tool):
    def getCC1Name(self, type):
        """getCC1Name(type) -> name, use-cpp, is-cxx"""
        
        # FIXME: Get bool results from elsewhere.
        if type is Types.CType or type is Types.CHeaderType:
            return 'cc1',True,False
        elif type is Types.CTypeNoPP or type is Types.CHeaderNoPPType:
            return 'cc1',False,False
        elif type is Types.ObjCType or type is Types.ObjCHeaderType:
            return 'cc1obj',True,False
        elif type is Types.ObjCTypeNoPP or type is Types.ObjCHeaderNoPPType:
            return 'cc1obj',True,False
        elif type is Types.CXXType or type is Types.CXXHeaderType:
            return 'cc1plus',True,True
        elif type is Types.CXXTypeNoPP or type is Types.CXXHeaderNoPPType:
            return 'cc1plus',False,True
        elif type is Types.ObjCXXType or type is Types.ObjCXXHeaderType:
            return 'cc1objplus',True,True
        elif type is Types.ObjCXXTypeNoPP or type is Types.ObjCXXHeaderNoPPType:
            return 'cc1objplus',False,True
        else:
            raise ValueError,"Unexpected type for Darwin compile tool."
        
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
#        if (not arglist.getLastArg(arglist.parser.m_macosxVersionMinOption) and
#            not arglist.getLastArg(arglist.parser.m_iphoneosVersionMinOption)):
#            cmd_args.append('-mmacosx-version-min=' + 
#                            self.toolChain.getMacosxVersionMin())

        # FIXME: Remove faltivec
        # FIXME: Remove mno-fused-madd
        # FIXME: Remove mlong-branch
        # FIXME: Remove mlongcall
        # FIXME: Remove mcpu=G4
        # FIXME: Remove mcpu=G5

        if (arglist.getLastArg(arglist.parser.gOption) and
            not arglist.getLastArg(arglist.parser.f_noEliminateUnusedDebugSymbolsOption)):
            cmd_args.append('-feliminate-unused-debug-symbols')

    def addCC1OptionsArgs(self, cmd_args, arch, arglist, inputs, output_args, isCXX):
        # Derived from cc1_options spec.
        if (arglist.getLastArg(arglist.parser.fastOption) or
            arglist.getLastArg(arglist.parser.fastfOption) or
            arglist.getLastArg(arglist.parser.fastcpOption)):
            cmd_args.append('-O3')
            
        if (arglist.getLastArg(arglist.parser.pgOption) and
            arglist.getLastArg(arglist.parser.f_omitFramePointerOption)):
            raise Arguments.InvalidArgumentsError("-pg and -fomit-frame-pointer are incompatible")

        self.addCC1Args(cmd_args, arch, arglist)

        if not arglist.getLastArg(arglist.parser.QOption):
            cmd_args.append('-quiet')

        cmd_args.append('-dumpbase')
        cmd_args.append(self.getBaseInputName(inputs, arglist))

        arglist.addAllArgs(cmd_args, arglist.parser.dGroup)

        arglist.addAllArgs(cmd_args, arglist.parser.mGroup)
        arglist.addAllArgs(cmd_args, arglist.parser.aGroup)

        # FIXME: The goal is to use the user provided -o if that is
        # our final output, otherwise to drive from the original input
        # name. Find a clean way to go about this.
        if (arglist.getLastArg(arglist.parser.cOption) or
            arglist.getLastArg(arglist.parser.SOption)):            
            outputOpt = arglist.getLastArg(arglist.parser.oOption)
            if outputOpt:
                cmd_args.append('-auxbase-strip')
                cmd_args.append(arglist.getValue(outputOpt))
            else:
                cmd_args.append('-auxbase')
                cmd_args.append(self.getBaseInputStem(inputs, arglist))
        else:
            cmd_args.append('-auxbase')
            cmd_args.append(self.getBaseInputStem(inputs, arglist))

        arglist.addAllArgs(cmd_args, arglist.parser.gGroup)

        arglist.addAllArgs(cmd_args, arglist.parser.OOption)
        # FIXME: -Wall is getting some special treatment. Investigate.
        arglist.addAllArgs2(cmd_args, arglist.parser.WGroup, arglist.parser.pedanticGroup)
        arglist.addLastArg(cmd_args, arglist.parser.wOption)
        arglist.addAllArgs3(cmd_args, arglist.parser.stdOption, arglist.parser.ansiOption, arglist.parser.trigraphsOption)
        if arglist.getLastArg(arglist.parser.vOption):
            cmd_args.append('-version')
        if arglist.getLastArg(arglist.parser.pgOption):
            cmd_args.append('-p')
        arglist.addLastArg(cmd_args, arglist.parser.pOption)
        
        # ccc treats -fsyntax-only specially.
        arglist.addAllArgs2(cmd_args, arglist.parser.fGroup, 
                            arglist.parser.syntaxOnlyOption)

        arglist.addAllArgs(cmd_args, arglist.parser.undefOption)
        if arglist.getLastArg(arglist.parser.QnOption):
            cmd_args.append('-fno-ident')
         
        # FIXME: This isn't correct.
        #arglist.addLastArg(cmd_args, arglist.parser._helpOption)
        #arglist.addLastArg(cmd_args, arglist.parser._targetHelpOption)

        if output_args:
            cmd_args.extend(output_args)

        # FIXME: Still don't get what is happening here. Investigate.
        arglist.addAllArgs(cmd_args, arglist.parser._paramOption)

        if (arglist.getLastArg(arglist.parser.f_mudflapOption) or
            arglist.getLastArg(arglist.parser.f_mudflapthOption)):
            cmd_args.append('-fno-builtin')
            cmd_args.append('-fno-merge-constants')

        if arglist.getLastArg(arglist.parser.coverageOption):
            cmd_args.append('-fprofile-arcs')
            cmd_args.append('-ftest-coverage')

        if isCXX:
            cmd_args.append('-D__private_extern__=extern')

    def getBaseInputName(self, inputs, arglist):
        # FIXME: gcc uses a temporary name here when the base
        # input is stdin, but only in auxbase. Investigate.
        baseInputValue = arglist.getValue(inputs[0].baseInput)
        return os.path.basename(baseInputValue)
    
    def getBaseInputStem(self, inputs, arglist):
        return os.path.splitext(self.getBaseInputName(inputs, arglist))[0]

    def getOutputArgs(self, arglist, output, isCPP=False):
        if isinstance(output, Jobs.PipedJob):
            if isCPP:
                return []
            else:
                return ['-o', '-']
        elif output is None:
            return ['-o', '/dev/null']
        else:
            return arglist.render(output)

    def addCPPOptionsArgs(self, cmd_args, arch, arglist, inputs,
                          output_args, isCXX):
        # Derived from cpp_options.
        self.addCPPUniqueOptionsArgs(cmd_args, arch, arglist, inputs)
        
        cmd_args.extend(output_args)

        self.addCC1Args(cmd_args, arch, arglist)

        # NOTE: The code below has some commonality with cpp_options,
        # but in classic gcc style ends up sending things in different
        # orders. This may be a good merge candidate once we drop
        # pedantic compatibility.

        arglist.addAllArgs(cmd_args, arglist.parser.mGroup)
        arglist.addAllArgs3(cmd_args, arglist.parser.stdOption, 
                            arglist.parser.ansiOption, 
                            arglist.parser.trigraphsOption)
        arglist.addAllArgs2(cmd_args, arglist.parser.WGroup, 
                            arglist.parser.pedanticGroup)
        arglist.addLastArg(cmd_args, arglist.parser.wOption)

        # ccc treats -fsyntax-only specially.
        arglist.addAllArgs2(cmd_args, arglist.parser.fGroup, 
                            arglist.parser.syntaxOnlyOption)

        if (arglist.getLastArg(arglist.parser.gGroup) and
            not arglist.getLastArg(arglist.parser.g0Option) and
            not arglist.getLastArg(arglist.parser.f_noWorkingDirectoryOption)):
            cmd_args.append('-fworking-directory')

        arglist.addAllArgs(cmd_args, arglist.parser.OOption)
        arglist.addAllArgs(cmd_args, arglist.parser.undefOption)
        if arglist.getLastArg(arglist.parser.saveTempsOption):
            cmd_args.append('-fpch-preprocess')

    def addCPPUniqueOptionsArgs(self, cmd_args, arch, arglist, inputs):
        # Derived from cpp_unique_options.

        if (arglist.getLastArg(arglist.parser.COption) or
            arglist.getLastArg(arglist.parser.CCOption)):
            if not arglist.getLastArg(arglist.parser.EOption):
                raise Arguments.InvalidArgumentsError("-C or -CC is not supported without -E")
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

        arglist.addAllArgs(cmd_args, arglist.parser.iGroup)

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

    def addCPPArgs(self, cmd_args, arch, arglist):
        # Derived from cpp spec.

        if arglist.getLastArg(arglist.parser.staticOption):
            # The gcc spec is broken here, it refers to dynamic but
            # that has been translated. Start by being bug compatible.
            
            # if not arglist.getLastArg(arglist.parser.dynamicOption):
            cmd_args.append('-D__STATIC__')
        else:
            cmd_args.append('-D__DYNAMIC__')
        
        if arglist.getLastArg(arglist.parser.pthreadOption):
            cmd_args.append('-D_REENTRANT')
        
class Darwin_X86_PreprocessTool(Darwin_X86_CC1Tool):
    def __init__(self, toolChain):
        super(Darwin_X86_PreprocessTool, self).__init__('cpp',
                                                        (Tool.eFlagsPipedInput |
                                                         Tool.eFlagsPipedOutput))
        self.toolChain = toolChain
    
    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, arglist, linkingOutput):
        inputType = inputs[0].type
        assert not [i for i in inputs if i.type != inputType]

        cc1Name,usePP,isCXX = self.getCC1Name(inputType)

        cmd_args = ['-E']
        if (arglist.getLastArg(arglist.parser.traditionalOption) or
            arglist.getLastArg(arglist.parser.f_traditionalOption) or
            arglist.getLastArg(arglist.parser.traditionalCPPOption)):
            cmd_args.append('-traditional-cpp')

        output_args = self.getOutputArgs(arglist, output,
                                         isCPP=True)
        self.addCPPOptionsArgs(cmd_args, arch, arglist, inputs, 
                               output_args, isCXX)

        arglist.addAllArgs(cmd_args, arglist.parser.dGroup)

        jobs.addJob(Jobs.Command(self.toolChain.getProgramPath(cc1Name), 
                                 cmd_args))
    
class Darwin_X86_CompileTool(Darwin_X86_CC1Tool):
    def __init__(self, toolChain):
        super(Darwin_X86_CompileTool, self).__init__('cc1',
                                                     (Tool.eFlagsPipedInput |
                                                      Tool.eFlagsPipedOutput |
                                                      Tool.eFlagsIntegratedCPP))
        self.toolChain = toolChain

    def constructJob(self, phase, arch, jobs, inputs, 
                     output, outputType, arglist, linkingOutput):
        inputType = inputs[0].type
        assert not [i for i in inputs if i.type != inputType]

        cc1Name,usePP,isCXX = self.getCC1Name(inputType)

        cmd_args = []
        if (arglist.getLastArg(arglist.parser.traditionalOption) or
            arglist.getLastArg(arglist.parser.f_traditionalOption)):
            raise Arguments.InvalidArgumentsError("-traditional is not supported without -E")

        if outputType is Types.PCHType:
            output_args = []
        else:
            output_args = self.getOutputArgs(arglist, output)
    
        # There is no need for this level of compatibility, but it
        # makes diffing easier.
        if (not arglist.getLastArg(arglist.parser.syntaxOnlyOption) and
            not arglist.getLastArg(arglist.parser.SOption)):
            early_output_args, end_output_args = [], output_args
        else:
            early_output_args, end_output_args = output_args, []

        if usePP:
            self.addCPPUniqueOptionsArgs(cmd_args, arch, arglist, inputs)
            self.addCC1OptionsArgs(cmd_args, arch, arglist, inputs,
                                   early_output_args, isCXX)
            cmd_args.extend(end_output_args)
        else:
            cmd_args.append('-fpreprocessed')
            
            # FIXME: There is a spec command to remove
            # -fpredictive-compilation args here. Investigate.

            for input in inputs:
                if isinstance(input.source, Jobs.PipedJob):
                    cmd_args.append('-')
                else:
                    cmd_args.extend(arglist.renderAsInput(input.source))

            self.addCC1OptionsArgs(cmd_args, arch, arglist, inputs, 
                                   early_output_args, isCXX)
            cmd_args.extend(end_output_args)

        if outputType is Types.PCHType:
            assert output is not None and not isinstance(output, Jobs.PipedJob)

            cmd_args.append('-o')
            # NOTE: gcc uses a temp .s file for this, but there
            # doesn't seem to be a good reason.
            cmd_args.append('/dev/null')
            
            cmd_args.append('--output-pch=')
            cmd_args.append(arglist.getValue(output))            
            
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
                raise Arguments.InvalidArgumentsError("invalid version number %r" % version)
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
        arglist.addAllArgs(cmd_args, arglist.parser.staticOption)
        if not arglist.getLastArg(arglist.parser.staticOption):
            cmd_args.append('-dynamic')
        if arglist.getLastArg(arglist.parser.f_gnuRuntimeOption):
            # FIXME: Replace -lobjc in forward args with
            # -lobjc-gnu. How do we wish to handle such things?
            pass

        if not arglist.getLastArg(arglist.parser.dynamiclibOption):
            if arglist.getLastArg(arglist.parser.force_cpusubtype_ALLOption):
                self.addDarwinArch(cmd_args, arch, arglist)
                cmd_args.append('-force_cpusubtype_ALL')
            else:
                self.addDarwinSubArch(cmd_args, arch, arglist)
        
            if arglist.getLastArg(arglist.parser.bundleOption):
                cmd_args.append('-bundle')
            arglist.addAllArgsTranslated(cmd_args, arglist.parser.bundle_loaderOption,
                                         '-bundle_loader')
            arglist.addAllArgs(cmd_args, arglist.parser.client_nameOption)
            if arglist.getLastArg(arglist.parser.compatibility_versionOption):
                # FIXME: Where should diagnostics go?
                print >>sys.stderr, "-compatibility_version only allowed with -dynamiclib"
                sys.exit(1)
            if arglist.getLastArg(arglist.parser.current_versionOption):
                print >>sys.stderr, "-current_version only allowed with -dynamiclib"
                sys.exit(1)
            if arglist.getLastArg(arglist.parser.force_flat_namespaceOption):
                cmd_args.append('-force_flat_namespace')
            if arglist.getLastArg(arglist.parser.install_nameOption):
                print >>sys.stderr, "-install_name only allowed with -dynamiclib"
                sys.exit(1)
            arglist.addLastArg(cmd_args, arglist.parser.keep_private_externsOption)
            arglist.addLastArg(cmd_args, arglist.parser.private_bundleOption)
        else:
            cmd_args.append('-dylib')
            if arglist.getLastArg(arglist.parser.bundleOption):
                print >>sys.stderr, "-bundle not allowed with -dynamiclib"
                sys.exit(1)
            if arglist.getLastArg(arglist.parser.bundle_loaderOption):
                print >>sys.stderr, "-bundle_loader not allowed with -dynamiclib"
                sys.exit(1)
            if arglist.getLastArg(arglist.parser.client_nameOption):
                print >>sys.stderr, "-client_name not allowed with -dynamiclib"
                sys.exit(1)
            arglist.addAllArgsTranslated(cmd_args, arglist.parser.compatibility_versionOption,
                                         '-dylib_compatibility_version')
            arglist.addAllArgsTranslated(cmd_args, arglist.parser.current_versionOption,
                                         '-dylib_current_version')
 
            if arglist.getLastArg(arglist.parser.force_cpusubtype_ALLOption):
                self.addDarwinArch(cmd_args, arch, arglist)
                # NOTE: We don't add -force_cpusubtype_ALL on this path. Ok.
            else:
                self.addDarwinSubArch(cmd_args, arch, arglist)
        
            if arglist.getLastArg(arglist.parser.force_flat_namespaceOption):
                print >>sys.stderr, "-force_flat_namespace not allowed with -dynamiclib"
                sys.exit(1)

            arglist.addAllArgsTranslated(cmd_args, arglist.parser.install_nameOption,
                                         '-dylib_install_name')

            if arglist.getLastArg(arglist.parser.keep_private_externsOption):
                print >>sys.stderr, "-keep_private_externs not allowed with -dynamiclib"
                sys.exit(1)
            if arglist.getLastArg(arglist.parser.private_bundleOption):
                print >>sys.stderr, "-private_bundle not allowed with -dynamiclib"
                sys.exit(1)

        if arglist.getLastArg(arglist.parser.all_loadOption):
            cmd_args.append('-all_load')

        arglist.addAllArgsTranslated(cmd_args, arglist.parser.allowable_clientOption,
                                     '-allowable_client')

        if arglist.getLastArg(arglist.parser.bind_at_loadOption):
            cmd_args.append('-bind_at_load')

        if arglist.getLastArg(arglist.parser.dead_stripOption):
            cmd_args.append('-dead_strip')
        
        if arglist.getLastArg(arglist.parser.no_dead_strip_inits_and_termsOption):
            cmd_args.append('-no_dead_strip_inits_and_terms')
        
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.dylib_fileOption,
                                     '-dylib_file')

        if arglist.getLastArg(arglist.parser.dynamicOption):
            cmd_args.append('-dynamic')

        arglist.addAllArgsTranslated(cmd_args, arglist.parser.exported_symbols_listOption,
                                     '-exported_symbols_list')

        if arglist.getLastArg(arglist.parser.flat_namespaceOption):
            cmd_args.append('-flat_namespace')

        arglist.addAllArgs(cmd_args, arglist.parser.headerpad_max_install_namesOption)
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.image_baseOption,
                                     '-image_base')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.initOption,
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
        
        if arglist.getLastArg(arglist.parser.multi_moduleOption):
            cmd_args.append('-multi_module')
        
        if arglist.getLastArg(arglist.parser.single_moduleOption):
            cmd_args.append('-single_module')

        arglist.addAllArgsTranslated(cmd_args, arglist.parser.multiply_definedOption,
                                     '-multiply_defined')

        arglist.addAllArgsTranslated(cmd_args, arglist.parser.multiply_defined_unusedOption,
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
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.segaddrOption,
                                     '-segaddr')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.segs_read_only_addrOption,
                                     '-segs_read_only_addr')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.segs_read_write_addrOption,
                                     '-segs_read_write_addr')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.seg_addr_tableOption,
                                     '-seg_addr_table')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.seg_addr_table_filenameOption,
                                     '-seg_addr_table_filename')
        arglist.addAllArgs(cmd_args, arglist.parser.sub_libraryOption)
        arglist.addAllArgs(cmd_args, arglist.parser.sub_umbrellaOption)
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.isysrootOption,
                                     '-syslibroot')
        arglist.addLastArg(cmd_args, arglist.parser.twolevel_namespaceOption)
        arglist.addLastArg(cmd_args, arglist.parser.twolevel_namespace_hintsOption)
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.umbrellaOption,
                                     '-umbrella')
        arglist.addAllArgs(cmd_args, arglist.parser.undefinedOption)
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.unexported_symbols_listOption,
                                     '-unexported_symbols_list')
        arglist.addAllArgsTranslated(cmd_args, arglist.parser.weak_reference_mismatchesOption,
                                     '-weak_reference_mismatches')
        
        if not arglist.getLastArg(arglist.parser.weak_reference_mismatchesOption):
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
                     output, outputType, arglist, linkingOutput):
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
        arglist.addLastArg(cmd_args, arglist.parser.dGroup)
        arglist.addLastArg(cmd_args, arglist.parser.tOption)
        arglist.addLastArg(cmd_args, arglist.parser.ZOption)
        arglist.addAllArgs(cmd_args, arglist.parser.uGroup)
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
            if arglist.getLastArg(arglist.parser.dynamiclibOption):
                # Derived from darwin_dylib1 spec.
                if arglist.getLastArg(arglist.parser.m_iphoneosVersionMinOption):
                    cmd_args.append('-ldylib1.o')
                else:
                    if macosxVersion < (10,5):
                        cmd_args.append('-ldylib1.o')
                    else:
                        cmd_args.append('-ldylib1.10.5.o')
            else:
                if arglist.getLastArg(arglist.parser.bundleOption):
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
                        cmd_args.append(self.toolChain.getFilePath('crt3.o'))

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

        if linkingOutput:
            cmd_args.append('-arch_multiple')
            cmd_args.append('-final_output')
            cmd_args.append(arglist.getValue(linkingOutput))

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

        if (arglist.getLastArg(arglist.parser.gGroup) and
            not arglist.getLastArg(arglist.parser.gstabsOption) and
            not arglist.getLastArg(arglist.parser.g0Option)):
            # FIXME: This is gross, but matches gcc. The test only
            # considers the suffix (not the -x type), and then only of the
            # first input.
            inputSuffix = os.path.splitext(arglist.getValue(inputs[0].baseInput))[1]        
            if inputSuffix in ('.c','.cc','.C','.cpp','.cp',
                               '.c++','.cxx','.CPP','.m','.mm'):
                jobs.addJob(Jobs.Command('dsymutil', 
                                         arglist.renderAsInput(output)))

class LipoTool(Tool):
    def __init__(self):
        super(LipoTool, self).__init__('lipo')

    def constructJob(self, phase, arch, jobs, inputs,
                     output, outputType, arglist, linkingOutput):
        assert outputType is Types.ImageType

        cmd_args = ['-create']
        cmd_args.extend(arglist.render(output))
        for input in inputs:
            cmd_args.extend(arglist.renderAsInput(input.source))
        jobs.addJob(Jobs.Command('lipo', cmd_args))
