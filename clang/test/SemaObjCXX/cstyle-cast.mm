// RUN: clang-cc -fsyntax-only -verify %s

@protocol P @end
@interface I @end

int main() {
  void *cft;
  id oct = (id)cft;

  Class ccct;
  ccct = (Class)cft;

  I* iict = (I*)cft;

  id<P> pid = (id<P>)cft;

  I<P> *ip = (I<P>*)cft;
  
}

