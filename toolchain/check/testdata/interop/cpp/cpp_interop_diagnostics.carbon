// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
//
// ============================================================================
// Enhanced C++ Interop Diagnostics (from carbon-interop)
// Based on carbon-interop/src/diagnostics.carbon
// Provides C++ source location mapping and enhanced error reporting
// Addresses GitHub Issue #5245: C++ locations in Carbon diagnostics
// ============================================================================

package CarbonCppInterop api;

import Std.String;
import Std.Array;
import Std.Optional;
import Std.HashMap;
import Std.Memory;

// Enhanced C++ source location tracking
class CppSourceLocation {
    var file_path: String;
    var line_number: i32;
    var column_number: i32;
    var offset_in_file: i64;
    var include_depth: i32;
    var macro_expansion_loc: Optional(CppSourceLocation);
    
    fn Create(file_path: String, line: i32, column: i32, offset: i64) -> Self {
        return {
            .file_path = file_path,
            .line_number = line,
            .column_number = column,
            .offset_in_file = offset,
            .include_depth = 0,
            .macro_expansion_loc = None(CppSourceLocation)
        };
    }
    
    fn CreateWithIncludeDepth(file_path: String, line: i32, column: i32, offset: i64, depth: i32) -> Self {
        var location: Self = Self.Create(file_path, line, column, offset);
        location.include_depth = depth;
        return location;
    }
    
    fn ToString[self: Self]() -> String {
        var result: String = self.file_path + ":" + ToString(self.line_number) + ":" + ToString(self.column_number);
        
        // Add include context if nested
        if (self.include_depth > 0) {
            result = result + " (in file included " + ToString(self.include_depth) + " times)";
        }
        
        // Add macro expansion context
        if (let macro_loc: CppSourceLocation = self.macro_expansion_loc) {
            result = result + "\n  expanded from: " + macro_loc.ToString();
        }
        
        return result;
    }
    
    fn GetDisplayPath[self: Self]() -> String {
        // Simplify path for display (remove common prefixes)
        var path: String = self.file_path;
        // Would implement path simplification logic
        return path;
    }
}

// Maps Carbon ImportIRInstId to C++ source locations with context
class CppLocationMapper {
    var location_mapping: HashMap(ImportIRInstId, CppSourceLocation);
    var source_files: Array(String);
    var include_hierarchy: HashMap(String, Array(String));
    var compilation_database: HashMap(String, CppCompilationInfo);
    
    fn Create() -> Self {
        return {
            .location_mapping = HashMap(ImportIRInstId, CppSourceLocation)(),
            .source_files = Array(String)(),
            .include_hierarchy = HashMap(String, Array(String))(),
            .compilation_database = HashMap(String, CppCompilationInfo)()
        };
    }
    
    // Enhanced location registration with include tracking
    fn RegisterCppLocation[addr self: Self](import_id: ImportIRInstId, cpp_location: CppSourceLocation) {
        self.location_mapping.Insert(import_id, cpp_location);
        
        // Track unique source files
        if (not self.IsFileTracked(cpp_location.file_path)) {
            self.source_files.Push(cpp_location.file_path);
        }
        
        // Build include hierarchy
        self.UpdateIncludeHierarchy(cpp_location);
    }
    
    // Register include relationship
    fn RegisterInclude[addr self: Self](including_file: String, included_file: String) {
        if (let includes: Array(String) = self.include_hierarchy.Get(including_file)) {
            includes.Push(included_file);
        } else {
            var new_includes: Array(String) = Array(String)();
            new_includes.Push(included_file);
            self.include_hierarchy.Insert(including_file, new_includes);
        }
    }
    
    // Enhanced location retrieval with context
    fn GetCppLocation[self: Self](import_id: ImportIRInstId) -> Optional(CppSourceLocation) {
        return self.location_mapping.Get(import_id);
    }
    
    // Get enhanced location with include context
    fn GetCppLocationWithContext[self: Self](import_id: ImportIRInstId) -> Optional(CppLocationContext) {
        if (let location: CppSourceLocation = self.GetCppLocation(import_id)) {
            var context: CppLocationContext = CppLocationContext.Create(location);
            
            // Add include chain
            context.include_chain = self.GetIncludeChain(location.file_path);
            
            // Add compilation info
            if (let comp_info: CppCompilationInfo = self.compilation_database.Get(location.file_path)) {
                context.compilation_info = Some(comp_info);
            }
            
            return Some(context);
        }
        return None(CppLocationContext);
    }
    
    fn GetSourceFiles[self: Self]() -> Array(String) {
        return self.source_files;
    }
    
    // Private helper methods
    fn IsFileTracked[self: Self](file_path: String) -> Bool {
        for (tracked_file: String in self.source_files) {
            if (tracked_file == file_path) {
                return True;
            }
        }
        return False;
    }
    
    fn UpdateIncludeHierarchy[addr self: Self](location: CppSourceLocation) {
        // Would analyze include depth and update hierarchy
        // Implementation would integrate with Clang's include tracking
    }
    
    fn GetIncludeChain[self: Self](file_path: String) -> Array(String) {
        var chain: Array(String) = Array(String)();
        // Would build complete include chain
        return chain;
    }
}

// Enhanced diagnostic with complete C++ context
class InteropDiagnostic {
    var carbon_message: String;
    var carbon_location: Optional(SourceLocation);
    var cpp_location_context: Optional(CppLocationContext);
    var diagnostic_kind: DiagnosticKind;
    var related_locations: Array(CppSourceLocation);
    var suggested_fixes: Array(DiagnosticFix);
    var performance_note: Optional(String);
    
    fn Create(message: String, kind: DiagnosticKind) -> Self {
        return {
            .carbon_message = message,
            .carbon_location = None(SourceLocation),
            .cpp_location_context = None(CppLocationContext),
            .diagnostic_kind = kind,
            .related_locations = Array(CppSourceLocation)(),
            .suggested_fixes = Array(DiagnosticFix)(),
            .performance_note = None(String)
        };
    }
    
    fn WithCarbonLocation[self: Self](location: SourceLocation) -> Self {
        var result: Self = self;
        result.carbon_location = Some(location);
        return result;
    }
    
    fn WithCppLocationContext[self: Self](context: CppLocationContext) -> Self {
        var result: Self = self;
        result.cpp_location_context = Some(context);
        return result;
    }
    
    fn AddSuggestedFix[addr self: Self](fix: DiagnosticFix) {
        self.suggested_fixes.Push(fix);
    }
    
    fn AddPerformanceNote[addr self: Self](note: String) {
        self.performance_note = Some(note);
    }
    
    // Generate comprehensive diagnostic message with all context
    fn FormatDiagnostic[self: Self]() -> String {
        var result: String = "";
        
        // Primary location (prefer C++ with context)
        if (let cpp_context: CppLocationContext = self.cpp_location_context) {
            result = cpp_context.location.ToString() + ": " + self.diagnostic_kind.ToString() + ": " + self.carbon_message;
            
            // Add include context
            if (cpp_context.include_chain.Size() > 0) {
                result = result + "\n  Include chain:";
                for (include_file: String in cpp_context.include_chain) {
                    result = result + "\n    " + include_file;
                }
            }
            
            // Add compilation context
            if (let comp_info: CppCompilationInfo = cpp_context.compilation_info) {
                result = result + "\n  Compiled with: " + comp_info.ToString();
            }
        } else if (let carbon_loc: SourceLocation = self.carbon_location) {
            result = carbon_loc.ToString() + ": " + self.diagnostic_kind.ToString() + ": " + self.carbon_message;
        } else {
            result = self.diagnostic_kind.ToString() + ": " + self.carbon_message;
        }
        
        // Add related C++ locations
        for (related: CppSourceLocation in self.related_locations) {
            result = result + "\n  note: related C++ location: " + related.ToString();
        }
        
        // Add suggested fixes
        for (fix: DiagnosticFix in self.suggested_fixes) {
            result = result + "\n  " + fix.ToString();
        }
        
        // Add performance note if applicable
        if (let perf_note: String = self.performance_note) {
            result = result + "\n  performance: " + perf_note;
        }
        
        return result;
    }
}

// Enhanced diagnostic context with complete C++ information
class CppLocationContext {
    var location: CppSourceLocation;
    var include_chain: Array(String);
    var compilation_info: Optional(CppCompilationInfo);
    var macro_expansion_stack: Array(CppSourceLocation);
    
    fn Create(location: CppSourceLocation) -> Self {
        return {
            .location = location,
            .include_chain = Array(String)(),
            .compilation_info = None(CppCompilationInfo),
            .macro_expansion_stack = Array(CppSourceLocation)()
        };
    }
}

// C++ compilation information for enhanced diagnostics
class CppCompilationInfo {
    var compiler_version: String;
    var compilation_flags: Array(String);
    var include_paths: Array(String);
    var defines: Array(String);
    var language_standard: String;
    
    fn Create(compiler: String, standard: String) -> Self {
        return {
            .compiler_version = compiler,
            .compilation_flags = Array(String)(),
            .include_paths = Array(String)(),
            .defines = Array(String)(),
            .language_standard = standard
        };
    }
    
    fn ToString[self: Self]() -> String {
        return self.compiler_version + " -std=" + self.language_standard;
    }
}

// Diagnostic fix suggestions
class DiagnosticFix {
    var description: String;
    var fix_kind: FixKind;
    var replacement_text: Optional(String);
    var file_path: Optional(String);
    var location: Optional(CppSourceLocation);
    
    fn Create(description: String, kind: FixKind) -> Self {
        return {
            .description = description,
            .fix_kind = kind,
            .replacement_text = None(String),
            .file_path = None(String),
            .location = None(CppSourceLocation)
        };
    }
    
    fn ToString[self: Self]() -> String {
        return "fix: " + self.description;
    }
}

// Types of diagnostic fixes
choice FixKind {
    IncludeHeader(),
    ChangeDeclaration(),
    AddTypeAnnotation(),
    PerformanceOptimization()
}

// Diagnostic severity levels with Carbon integration
choice DiagnosticKind {
    Error(),
    Warning(),
    Note(),
    Info(),
    PerformanceWarning()
}

impl DiagnosticKind {
    fn ToString[self: Self]() -> String {
        return match (self) {
            case .Error() => { return "error"; }
            case .Warning() => { return "warning"; }
            case .Note() => { return "note"; }
            case .Info() => { return "info"; }
            case .PerformanceWarning() => { return "performance warning"; }
        };
    }
}

// Enhanced Clang diagnostic converter with complete context
class ClangDiagnosticConverter {
    var location_mapper: CppLocationMapper;
    var active_cpp_imports: Array(String);
    var diagnostic_filters: Array(DiagnosticFilter);
    var performance_threshold_ns: f64;
    
    fn Create() -> Self {
        return {
            .location_mapper = CppLocationMapper.Create(),
            .active_cpp_imports = Array(String)(),
            .diagnostic_filters = Array(DiagnosticFilter)(),
            .performance_threshold_ns = 10.0  // Warn if >10ns overhead
        };
    }
    
    // Enhanced Clang diagnostic conversion with full context
    fn ConvertClangDiagnostic[addr self: Self](
        clang_message: String,
        clang_file: String,
        clang_line: i32,
        clang_column: i32,
        severity: DiagnosticKind,
        include_stack: Array(String)
    ) -> InteropDiagnostic {
        var cpp_location: CppSourceLocation = CppSourceLocation.Create(
            clang_file, clang_line, clang_column, 0
        );
        
        var context: CppLocationContext = CppLocationContext.Create(cpp_location);
        context.include_chain = include_stack;
        
        var diagnostic: InteropDiagnostic = InteropDiagnostic.Create(
            "C++ interop: " + clang_message, severity
        );
        
        diagnostic = diagnostic.WithCppLocationContext(context);
        
        // Add suggested fixes based on common patterns
        self.AddCommonFixes(diagnostic, clang_message);
        
        return diagnostic;
    }
    
    // Generate performance diagnostic
    fn CreatePerformanceDiagnostic[self: Self](
        operation: String,
        overhead_ns: f64,
        location: CppSourceLocation
    ) -> InteropDiagnostic {
        if (overhead_ns > self.performance_threshold_ns) {
            var message: String = operation + " operation has " + ToString(overhead_ns) + "ns overhead";
            var diagnostic: InteropDiagnostic = InteropDiagnostic.Create(
                message, DiagnosticKind.PerformanceWarning()
            );
            
            var context: CppLocationContext = CppLocationContext.Create(location);
            return diagnostic.WithCppLocationContext(context);
        }
        
        return InteropDiagnostic.Create("", DiagnosticKind.Info());
    }
    
    fn RegisterCppImport[addr self: Self](import_path: String, import_id: ImportIRInstId) {
        self.active_cpp_imports.Push(import_path);
    }
    
    // Private helper for common diagnostic fixes
    fn AddCommonFixes[self: Self](diagnostic: InteropDiagnostic, message: String) {
        // Would analyze common error patterns and suggest fixes
        // Implementation would use pattern matching on Clang messages
    }
}

// Diagnostic filtering for noise reduction
class DiagnosticFilter {
    var filter_pattern: String;
    var action: FilterAction;
    var replacement_message: Optional(String);
    
    fn Create(pattern: String, action: FilterAction) -> Self {
        return {
            .filter_pattern = pattern,
            .action = action,
            .replacement_message = None(String)
        };
    }
}

choice FilterAction {
    Suppress(),
    Downgrade(),
    Replace()
}

// Opaque types for integration
type SourceLocation = Void;  // Would be Carbon's SourceLocation
type ImportIRInstId = i32;   // Would be Carbon's ImportIRInstId

// Utility function
fn ToString(value: i32) -> String {
    // Would use Carbon's integer to string conversion
    return "42";  // Placeholder
}