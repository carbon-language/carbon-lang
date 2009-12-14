// RUN: clang -cc1 -rewrite-objc %s -o -

@interface Foo @end
@interface GARF @end

void foo() {
  @try  { TRY(); } 
  @catch (...) { SPLATCH(); @throw; }
}

int main()
{

  @try  {
     MYTRY();
  }

  @catch (Foo* localException) {
     MYCATCH();
     @throw;
  }
  
  // no catch clause
  @try { } 
  @finally { }
}

