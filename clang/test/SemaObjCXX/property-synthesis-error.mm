// RUN: %clang_cc1 -fsyntax-only -verify %s
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
 TCPPObject& operator=(const TCPPObject& inObj);
private:
 void* fData;
};

@interface MyDocument
{
@private
 TCPPObject _cppObject;
 TCPPObject _ncppObject;
}
@property (assign, readwrite) const TCPPObject& cppObject;
@property (assign, readwrite, nonatomic) const TCPPObject& ncppObject;
@end

@implementation MyDocument

@synthesize cppObject = _cppObject; // expected-warning {{atomic property of type 'const TCPPObject &' synthesizing setter using non-trivial assignmentoperator}}
@synthesize ncppObject = _ncppObject;

@end
