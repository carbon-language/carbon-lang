// RUN: clang -fsyntax-only -verify %s

typedef struct Protocol Protocol;

@protocol fproto;

@protocol p1 
@end

@class cl;

int main()
{
	Protocol *proto = @protocol(p1);
        Protocol *fproto = @protocol(fproto);
}

