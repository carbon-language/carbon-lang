// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <unordered_map>

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Diagnostic.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

/**
 * Enhanced C++ Interop Diagnostics System
 * 
 * Addresses GitHub Issue #5245: C++ locations in Carbon diagnostics
 * 
 * This system provides precise mapping between Carbon diagnostic locations
 * and their corresponding C++ source locations, enabling developers to
 * quickly identify and fix interop issues.
 * 
 * Key Features:
 * - Precise C++ source location tracking
 * - Include hierarchy analysis  
 * - Macro expansion context
 * - Performance overhead reporting
 * - Suggested fixes for common interop issues
 */

// Enhanced C++ source location with complete context
struct CppSourceLocationInfo {
    std::string file_path;
    uint32_t line_number;
    uint32_t column_number;
    uint64_t offset_in_file;
    uint32_t include_depth;
    std::optional<CppSourceLocationInfo> macro_expansion_loc;
    std::vector<std::string> include_chain;
    
    CppSourceLocationInfo(std::string file, uint32_t line, uint32_t column, uint64_t offset)
        : file_path(std::move(file)), line_number(line), column_number(column), 
          offset_in_file(offset), include_depth(0) {}
    
    std::string ToString() const;
    std::string GetDisplayPath() const;
};

// C++ compilation context for enhanced diagnostics
struct CppCompilationContext {
    std::string compiler_version;
    std::vector<std::string> compilation_flags;
    std::vector<std::string> include_paths;
    std::vector<std::string> defines;
    std::string language_standard;
    
    CppCompilationContext(std::string compiler, std::string standard)
        : compiler_version(std::move(compiler)), language_standard(std::move(standard)) {}
    
    std::string ToString() const;
};

// Enhanced diagnostic with C++ context
class InteropDiagnosticInfo {
public:
    enum class Kind {
        Error,
        Warning, 
        Note,
        Info,
        PerformanceWarning
    };
    
    enum class FixKind {
        IncludeHeader,
        ChangeDeclaration,
        AddTypeAnnotation,
        PerformanceOptimization
    };
    
    struct Fix {
        std::string description;
        FixKind kind;
        std::optional<std::string> replacement_text;
        std::optional<std::string> file_path;
        std::optional<CppSourceLocationInfo> location;
        
        Fix(std::string desc, FixKind k) : description(std::move(desc)), kind(k) {}
        std::string ToString() const { return "fix: " + description; }
    };
    
private:
    std::string carbon_message_;
    std::optional<SemIR::LocId> carbon_location_;
    std::optional<CppSourceLocationInfo> cpp_location_;
    Kind diagnostic_kind_;
    std::vector<CppSourceLocationInfo> related_locations_;
    std::vector<Fix> suggested_fixes_;
    std::optional<std::string> performance_note_;
    std::optional<CppCompilationContext> compilation_context_;
    
public:
    InteropDiagnosticInfo(std::string message, Kind kind)
        : carbon_message_(std::move(message)), diagnostic_kind_(kind) {}
    
    InteropDiagnosticInfo& WithCarbonLocation(SemIR::LocId location);
    InteropDiagnosticInfo& WithCppLocation(CppSourceLocationInfo cpp_location);
    InteropDiagnosticInfo& WithCompilationContext(CppCompilationContext context);
    
    void AddRelatedLocation(CppSourceLocationInfo location);
    void AddSuggestedFix(Fix fix);
    void AddPerformanceNote(std::string note);
    
    std::string FormatDiagnostic() const;
    std::string GetKindString() const;
    
    // Getters
    const std::string& GetMessage() const { return carbon_message_; }
    Kind GetKind() const { return diagnostic_kind_; }
    const std::optional<CppSourceLocationInfo>& GetCppLocation() const { return cpp_location_; }
    const std::vector<Fix>& GetSuggestedFixes() const { return suggested_fixes_; }
};

// Maps Carbon ImportIRInstId to C++ source locations with full context
class CppLocationMapper {
private:
    std::unordered_map<SemIR::ImportIRInstId, CppSourceLocationInfo> location_mapping_;
    std::vector<std::string> source_files_;
    std::unordered_map<std::string, std::vector<std::string>> include_hierarchy_;
    std::unordered_map<std::string, CppCompilationContext> compilation_database_;
    
public:
    // Register C++ location with complete context
    void RegisterCppLocation(SemIR::ImportIRInstId import_id, 
                           const CppSourceLocationInfo& cpp_location);
    
    // Register include relationship for hierarchy tracking
    void RegisterInclude(const std::string& including_file, 
                        const std::string& included_file);
    
    // Register compilation context
    void RegisterCompilationContext(const std::string& file_path,
                                  CppCompilationContext context);
    
    // Retrieve location with full context
    std::optional<CppSourceLocationInfo> GetCppLocation(SemIR::ImportIRInstId import_id) const;
    
    // Get enhanced location with complete context
    std::optional<CppSourceLocationInfo> GetCppLocationWithContext(SemIR::ImportIRInstId import_id) const;
    
    // Get all tracked source files
    const std::vector<std::string>& GetSourceFiles() const { return source_files_; }
    
    // Get include chain for a file
    std::vector<std::string> GetIncludeChain(const std::string& file_path) const;
    
private:
    bool IsFileTracked(const std::string& file_path) const;
    void UpdateIncludeHierarchy(const CppSourceLocationInfo& location);
};

// Enhanced Clang diagnostic converter with comprehensive context
class ClangDiagnosticConverter {
private:
    std::unique_ptr<CppLocationMapper> location_mapper_;
    std::vector<std::string> active_cpp_imports_;
    double performance_threshold_ns_;
    
public:
    ClangDiagnosticConverter();
    ~ClangDiagnosticConverter();
    
    // Convert Clang diagnostic to Carbon diagnostic with full context
    InteropDiagnosticInfo ConvertClangDiagnostic(
        const std::string& clang_message,
        const clang::SourceLocation& clang_location,
        const clang::SourceManager& source_manager,
        clang::DiagnosticsEngine::Level severity,
        const std::vector<std::string>& include_stack = {}
    );
    
    // Create from Clang SourceLocation with full context extraction
    CppSourceLocationInfo CreateCppLocation(
        const clang::SourceLocation& clang_location,
        const clang::SourceManager& source_manager
    );
    
    // Generate performance diagnostic
    InteropDiagnosticInfo CreatePerformanceDiagnostic(
        const std::string& operation,
        double overhead_ns,
        const CppSourceLocationInfo& location
    );
    
    // Register C++ import for diagnostic context
    void RegisterCppImport(const std::string& import_path, SemIR::ImportIRInstId import_id);
    
    // Get location mapper for external access
    CppLocationMapper& GetLocationMapper() { return *location_mapper_; }
    
private:
    // Add common fix suggestions based on error patterns
    void AddCommonFixes(InteropDiagnosticInfo& diagnostic, const std::string& message);
    
    // Extract include chain from Clang SourceManager
    std::vector<std::string> ExtractIncludeChain(
        const clang::SourceLocation& location,
        const clang::SourceManager& source_manager
    );
    
    // Convert Clang diagnostic level to Carbon level
    InteropDiagnosticInfo::Kind ConvertDiagnosticLevel(clang::DiagnosticsEngine::Level level);
};

// Integration with Carbon's diagnostic system
class CarbonClangDiagnosticConsumerEnhanced : public clang::DiagnosticConsumer {
private:
    Context* context_;
    std::unique_ptr<ClangDiagnosticConverter> converter_;
    std::vector<InteropDiagnosticInfo> collected_diagnostics_;
    
public:
    explicit CarbonClangDiagnosticConsumerEnhanced(Context* context);
    ~CarbonClangDiagnosticConsumerEnhanced() override;
    
    // DiagnosticConsumer interface
    void HandleDiagnostic(clang::DiagnosticsEngine::Level diag_level,
                         const clang::Diagnostic& info) override;
    
    // Emit all collected diagnostics to Carbon's diagnostic system
    void EmitDiagnostics();
    
    // Get diagnostic converter for external access
    ClangDiagnosticConverter& GetConverter() { return *converter_; }
    
private:
    void EmitCarbonDiagnostic(const InteropDiagnosticInfo& diagnostic);
};

// Performance monitoring for interop operations
class InteropPerformanceMonitor {
private:
    std::unordered_map<std::string, double> operation_times_;
    std::unordered_map<std::string, uint64_t> operation_counts_;
    double warning_threshold_ns_;
    
public:
    explicit InteropPerformanceMonitor(double threshold_ns = 10.0)
        : warning_threshold_ns_(threshold_ns) {}
    
    void RecordOperation(const std::string& operation_name, double time_ns);
    void CheckPerformance(ClangDiagnosticConverter& converter, 
                         const CppSourceLocationInfo& location);
    
    double GetAverageTime(const std::string& operation_name) const;
    uint64_t GetOperationCount(const std::string& operation_name) const;
    
    std::vector<std::pair<std::string, double>> GetSlowOperations() const;
};

// Utility functions for diagnostic formatting
std::string FormatCppLocation(const CppSourceLocationInfo& location);
std::string FormatIncludeChain(const std::vector<std::string>& chain);
std::string FormatCompilationContext(const CppCompilationContext& context);

} // namespace Carbon::Check