//===- compiler.go - IR generator entry point -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main IR generator entry point, (*Compiler).Compile.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"bytes"
	"fmt"
	"go/token"
	"log"
	"sort"
	"strconv"
	"strings"

	llgobuild "llvm.org/llgo/build"
	"llvm.org/llgo/debug"
	"llvm.org/llvm/bindings/go/llvm"

	"llvm.org/llgo/third_party/go.tools/go/gccgoimporter"
	"llvm.org/llgo/third_party/go.tools/go/importer"
	"llvm.org/llgo/third_party/go.tools/go/loader"
	"llvm.org/llgo/third_party/go.tools/go/ssa"
	"llvm.org/llgo/third_party/go.tools/go/types"
)

type Module struct {
	llvm.Module
	Path       string
	ExportData []byte
	disposed   bool
}

func (m *Module) Dispose() {
	if m.disposed {
		return
	}
	m.Module.Dispose()
	m.disposed = true
}

///////////////////////////////////////////////////////////////////////////////

type CompilerOptions struct {
	// TargetTriple is the LLVM triple for the target.
	TargetTriple string

	// GenerateDebug decides whether debug data is
	// generated in the output module.
	GenerateDebug bool

	// DebugPrefixMaps is a list of mappings from source prefixes to
	// replacement prefixes, to be applied in debug info.
	DebugPrefixMaps []debug.PrefixMap

	// Logger is a logger used for tracing compilation.
	Logger *log.Logger

	// DumpSSA is a debugging option that dumps each SSA function
	// to stderr before generating code for it.
	DumpSSA bool

	// GccgoPath is the path to the gccgo binary whose libgo we read import
	// data from. If blank, the caller is expected to supply an import
	// path in ImportPaths.
	GccgoPath string

	// ImportPaths is the list of additional import paths
	ImportPaths []string

	// SanitizerAttribute is an attribute to apply to functions to enable
	// dynamic instrumentation using a sanitizer.
	SanitizerAttribute llvm.Attribute
}

type Compiler struct {
	opts       CompilerOptions
	dataLayout string
	pnacl      bool
}

func NewCompiler(opts CompilerOptions) (*Compiler, error) {
	compiler := &Compiler{opts: opts}
	if strings.ToLower(compiler.opts.TargetTriple) == "pnacl" {
		compiler.opts.TargetTriple = PNaClTriple
		compiler.pnacl = true
	}
	dataLayout, err := llvmDataLayout(compiler.opts.TargetTriple)
	if err != nil {
		return nil, err
	}
	compiler.dataLayout = dataLayout
	return compiler, nil
}

func (c *Compiler) Compile(filenames []string, importpath string) (m *Module, err error) {
	target := llvm.NewTargetData(c.dataLayout)
	compiler := &compiler{
		CompilerOptions: c.opts,
		dataLayout:      c.dataLayout,
		target:          target,
		pnacl:           c.pnacl,
		llvmtypes:       NewLLVMTypeMap(llvm.GlobalContext(), target),
	}
	return compiler.compile(filenames, importpath)
}

type compiler struct {
	CompilerOptions

	module     *Module
	dataLayout string
	target     llvm.TargetData
	fileset    *token.FileSet

	runtime   *runtimeInterface
	llvmtypes *llvmTypeMap
	types     *TypeMap

	// runtimetypespkg is the type-checked runtime/types.go file,
	// which is used for evaluating the types of runtime functions.
	runtimetypespkg *types.Package

	// pnacl is set to true if the target triple was originally
	// specified as "pnacl". This is necessary, as the TargetTriple
	// field will have been updated to the true triple used to
	// compile PNaCl modules.
	pnacl bool

	debug *debug.DIBuilder
}

func (c *compiler) logf(format string, v ...interface{}) {
	if c.Logger != nil {
		c.Logger.Printf(format, v...)
	}
}

func (c *compiler) addCommonFunctionAttrs(fn llvm.Value) {
	fn.AddTargetDependentFunctionAttr("disable-tail-calls", "true")
	fn.AddTargetDependentFunctionAttr("split-stack", "")
	if attr := c.SanitizerAttribute; attr != 0 {
		fn.AddFunctionAttr(attr)
	}
}

func (compiler *compiler) compile(filenames []string, importpath string) (m *Module, err error) {
	buildctx, err := llgobuild.ContextFromTriple(compiler.TargetTriple)
	if err != nil {
		return nil, err
	}

	initmap := make(map[*types.Package]gccgoimporter.InitData)
	var importer types.Importer
	if compiler.GccgoPath == "" {
		paths := append(append([]string{}, compiler.ImportPaths...), ".")
		importer = gccgoimporter.GetImporter(paths, initmap)
	} else {
		var inst gccgoimporter.GccgoInstallation
		err = inst.InitFromDriver(compiler.GccgoPath)
		if err != nil {
			return nil, err
		}
		importer = inst.GetImporter(compiler.ImportPaths, initmap)
	}

	impcfg := &loader.Config{
		Fset: token.NewFileSet(),
		TypeChecker: types.Config{
			Import: importer,
			Sizes:  compiler.llvmtypes,
		},
		Build: &buildctx.Context,
	}
	// Must use parseFiles, so we retain comments;
	// this is important for annotation processing.
	astFiles, err := parseFiles(impcfg.Fset, filenames)
	if err != nil {
		return nil, err
	}
	// If no import path is specified, then set the import
	// path to be the same as the package's name.
	if importpath == "" {
		importpath = astFiles[0].Name.String()
	}
	impcfg.CreateFromFiles(importpath, astFiles...)
	iprog, err := impcfg.Load()
	if err != nil {
		return nil, err
	}
	program := ssa.Create(iprog, ssa.BareInits)
	mainPkginfo := iprog.InitialPackages()[0]
	mainPkg := program.CreatePackage(mainPkginfo)

	// Create a Module, which contains the LLVM module.
	modulename := importpath
	compiler.module = &Module{Module: llvm.NewModule(modulename), Path: modulename}
	compiler.module.SetTarget(compiler.TargetTriple)
	compiler.module.SetDataLayout(compiler.dataLayout)

	// Create a new translation unit.
	unit := newUnit(compiler, mainPkg)

	// Create the runtime interface.
	compiler.runtime, err = newRuntimeInterface(compiler.module.Module, compiler.llvmtypes)
	if err != nil {
		return nil, err
	}

	mainPkg.Build()

	// Create a struct responsible for mapping static types to LLVM types,
	// and to runtime/dynamic type values.
	compiler.types = NewTypeMap(
		mainPkg,
		compiler.llvmtypes,
		compiler.module.Module,
		compiler.runtime,
		MethodResolver(unit),
	)

	if compiler.GenerateDebug {
		compiler.debug = debug.NewDIBuilder(
			types.Sizes(compiler.llvmtypes),
			compiler.module.Module,
			impcfg.Fset,
			compiler.DebugPrefixMaps,
		)
		defer compiler.debug.Destroy()
		defer compiler.debug.Finalize()
	}

	unit.translatePackage(mainPkg)
	compiler.processAnnotations(unit, mainPkginfo)

	if importpath == "main" {
		if err = compiler.createInitMainFunction(mainPkg, initmap); err != nil {
			return nil, fmt.Errorf("failed to create __go_init_main: %v", err)
		}
	} else {
		compiler.module.ExportData = compiler.buildExportData(mainPkg, initmap)
	}

	return compiler.module, nil
}

type byPriorityThenFunc []gccgoimporter.PackageInit

func (a byPriorityThenFunc) Len() int      { return len(a) }
func (a byPriorityThenFunc) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a byPriorityThenFunc) Less(i, j int) bool {
	switch {
	case a[i].Priority < a[j].Priority:
		return true
	case a[i].Priority > a[j].Priority:
		return false
	case a[i].InitFunc < a[j].InitFunc:
		return true
	default:
		return false
	}
}

func (c *compiler) buildPackageInitData(mainPkg *ssa.Package, initmap map[*types.Package]gccgoimporter.InitData) gccgoimporter.InitData {
	var inits []gccgoimporter.PackageInit
	for _, imp := range mainPkg.Object.Imports() {
		inits = append(inits, initmap[imp].Inits...)
	}
	sort.Sort(byPriorityThenFunc(inits))

	// Deduplicate init entries. We want to preserve the entry with the highest priority.
	// Normally a package's priorities will be consistent among its dependencies, but it is
	// possible for them to be different. For example, if a standard library test augments a
	// package which is a dependency of 'regexp' (which is imported by every test main package)
	// with additional dependencies, those dependencies may cause the package under test to
	// receive a higher priority than indicated by its init clause in 'regexp'.
	uniqinits := make([]gccgoimporter.PackageInit, len(inits))
	uniqinitpos := len(inits)
	uniqinitnames := make(map[string]struct{})
	for i, _ := range inits {
		init := inits[len(inits)-1-i]
		if _, ok := uniqinitnames[init.InitFunc]; !ok {
			uniqinitnames[init.InitFunc] = struct{}{}
			uniqinitpos--
			uniqinits[uniqinitpos] = init
		}
	}
	uniqinits = uniqinits[uniqinitpos:]

	ourprio := 1
	if len(uniqinits) != 0 {
		ourprio = uniqinits[len(uniqinits)-1].Priority + 1
	}

	if imp := mainPkg.Func("init"); imp != nil {
		impname := c.types.mc.mangleFunctionName(imp)
		uniqinits = append(uniqinits, gccgoimporter.PackageInit{mainPkg.Object.Name(), impname, ourprio})
	}

	return gccgoimporter.InitData{ourprio, uniqinits}
}

func (c *compiler) createInitMainFunction(mainPkg *ssa.Package, initmap map[*types.Package]gccgoimporter.InitData) error {
	initdata := c.buildPackageInitData(mainPkg, initmap)

	ftyp := llvm.FunctionType(llvm.VoidType(), nil, false)
	initMain := llvm.AddFunction(c.module.Module, "__go_init_main", ftyp)
	c.addCommonFunctionAttrs(initMain)
	entry := llvm.AddBasicBlock(initMain, "entry")

	builder := llvm.GlobalContext().NewBuilder()
	defer builder.Dispose()
	builder.SetInsertPointAtEnd(entry)

	for _, init := range initdata.Inits {
		initfn := c.module.Module.NamedFunction(init.InitFunc)
		if initfn.IsNil() {
			initfn = llvm.AddFunction(c.module.Module, init.InitFunc, ftyp)
		}
		builder.CreateCall(initfn, nil, "")
	}

	builder.CreateRetVoid()
	return nil
}

func (c *compiler) buildExportData(mainPkg *ssa.Package, initmap map[*types.Package]gccgoimporter.InitData) []byte {
	exportData := importer.ExportData(mainPkg.Object)
	b := bytes.NewBuffer(exportData)

	initdata := c.buildPackageInitData(mainPkg, initmap)
	b.WriteString("v1;\npriority ")
	b.WriteString(strconv.Itoa(initdata.Priority))
	b.WriteString(";\n")

	if len(initdata.Inits) != 0 {
		b.WriteString("init")
		for _, init := range initdata.Inits {
			b.WriteRune(' ')
			b.WriteString(init.Name)
			b.WriteRune(' ')
			b.WriteString(init.InitFunc)
			b.WriteRune(' ')
			b.WriteString(strconv.Itoa(init.Priority))
		}
		b.WriteString(";\n")
	}

	return b.Bytes()
}

// vim: set ft=go :
