// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fsyntax-only -verify %s
// rdar://9070460

class TCPPObject
{
public:
	TCPPObject(const TCPPObject& inObj);
	TCPPObject();
	~TCPPObject();
	
	TCPPObject& operator=(const TCPPObject& inObj)const ;

	void* Data();
	
private:
	void* fData;
};


typedef const TCPPObject& CREF_TCPPObject;

@interface TNSObject
@property (assign, readwrite, nonatomic) CREF_TCPPObject cppObjectNonAtomic;
@property (assign, readwrite) CREF_TCPPObject cppObjectAtomic;
@property (assign, readwrite, nonatomic) const TCPPObject& cppObjectDynamic;
@end


@implementation TNSObject

@synthesize cppObjectNonAtomic;
@synthesize cppObjectAtomic; // expected-warning{{atomic property of type 'CREF_TCPPObject' (aka 'const TCPPObject &') synthesizing setter using non-trivial assignment operator}}
@dynamic cppObjectDynamic;

- (const TCPPObject&) cppObjectNonAtomic
{
	return cppObjectNonAtomic;
}

- (void) setCppObjectNonAtomic: (const TCPPObject&)cppObject
{
	cppObjectNonAtomic = cppObject;
}
@end
