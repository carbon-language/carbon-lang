// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// rdar: //8550657

@interface NSArray @end

@interface NSMutableArray : NSArray @end

@interface MyClass
{
  NSMutableArray * _array;
}

@property (readonly) NSMutableArray * array;

@end

@interface MyClass ()

@property (readwrite) NSMutableArray * array;

@end

@implementation MyClass

@synthesize array=_array;

@end

int main(void)
{
  return 0;
}

// rdar://6137845
class TCPPObject
{
public:
 TCPPObject(const TCPPObject& inObj);
 TCPPObject();
 ~TCPPObject();
 TCPPObject& operator=(const TCPPObject& inObj); // expected-note {{'operator=' declared here}}
private:
 void* fData;
};

class Trivial
{
public:
 Trivial(const Trivial& inObj);
 Trivial();
 ~Trivial();
private:
 void* fData;
};

@interface MyDocument
{
@private
 TCPPObject _cppObject;
 TCPPObject _ncppObject;
 Trivial _tcppObject;
}
@property (assign, readwrite) const TCPPObject& cppObject;
@property (assign, readwrite, nonatomic) const TCPPObject& ncppObject;
@property (assign, readwrite) const Trivial& tcppObject;
@end

@implementation MyDocument

@synthesize cppObject = _cppObject; // expected-error {{atomic property of reference type 'const TCPPObject &' cannot have non-trivial assignment operator}}
@synthesize ncppObject = _ncppObject;

@synthesize tcppObject = _tcppObject;
@end
