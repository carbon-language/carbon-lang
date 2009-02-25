//===--- TemplateKinds.h - Enum values for C++ Template Kinds ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TemplateNameKind enum.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TEMPLATEKINDS_H
#define LLVM_CLANG_TEMPLATEKINDS_H

namespace clang {

/// \brief Specifies the kind of template name that an identifier refers to.
enum TemplateNameKind {
  /// The name does not refer to a template.
  TNK_Non_template = 0,
  /// The name refers to a function template or a set of overloaded
  /// functions that includes at least one function template.
  TNK_Function_template,
  /// The name refers to a class template.
  TNK_Class_template,
  /// The name referes to a template template parameter.
  TNK_Template_template_parm,
  /// The name is dependent and is known to be a template name based
  /// on syntax, e.g., "Alloc::template rebind<Other>".
  TNK_Dependent_template_name
};

}
#endif


