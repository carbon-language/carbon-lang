// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol NSTextViewDelegate;

@interface NSResponder @end

class AutoreleaseObject
{
public:
 AutoreleaseObject();
 ~AutoreleaseObject();


 AutoreleaseObject& operator=(NSResponder* inValue);
 AutoreleaseObject& operator=(const AutoreleaseObject& inValue);

 AutoreleaseObject(const AutoreleaseObject& inValue);

 operator NSResponder*() const;
};


void InvokeSaveFocus()
{
 AutoreleaseObject mResolvedFirstResponder;
 id<NSTextViewDelegate> Mydelegate;
 mResolvedFirstResponder = static_cast<NSResponder*>(Mydelegate);
}

