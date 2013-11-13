//===--- CommentToXML.h - Convert comments to XML representation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_COMMENTTOXML_H
#define LLVM_CLANG_INDEX_COMMENTTOXML_H

#include "clang/Basic/LLVM.h"

namespace clang {
class ASTContext;

namespace comments {
class FullComment;
class HTMLTagComment;
}

namespace index {
class SimpleFormatContext;

class CommentToXMLConverter {
  SimpleFormatContext *FormatContext;
  unsigned FormatInMemoryUniqueId;

public:
  CommentToXMLConverter() : FormatContext(0), FormatInMemoryUniqueId(0) {}

  void convertCommentToHTML(const comments::FullComment *FC,
                            SmallVectorImpl<char> &HTML,
                            const ASTContext &Context);

  void convertHTMLTagNodeToText(const comments::HTMLTagComment *HTC,
                                SmallVectorImpl<char> &Text,
                                const ASTContext &Context);

  void convertCommentToXML(const comments::FullComment *FC,
                           SmallVectorImpl<char> &XML,
                           const ASTContext &Context);
};

} // namespace index
} // namespace clang

#endif // LLVM_CLANG_INDEX_COMMENTTOXML_H

