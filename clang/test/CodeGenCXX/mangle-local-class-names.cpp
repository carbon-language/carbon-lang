// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// CHECK:  @_ZZ4FUNCvEN4SSSSC1ERKf
// CHECK: @_ZZ4FUNCvEN4SSSSC2E_0RKf
// CHECK:  @_ZZ4GORFfEN4SSSSC1ERKf
// CHECK: @_ZZ4GORFfEN4SSSSC2E_0RKf

void FUNC ()
{
  {
    float IVAR1 ;

    struct SSSS 
    {
      float bv;
      SSSS( const float& from): bv(from) { }
    };

    SSSS VAR1(IVAR1);
   }

   {
    float IVAR2 ;

    struct SSSS
    {
     SSSS( const float& from) {}
    };

    SSSS VAR2(IVAR2);
   }
}

void GORF (float IVAR1)
{
  {
    struct SSSS 
    {
      float bv;
      SSSS( const float& from): bv(from) { }
    };

    SSSS VAR1(IVAR1);
   }

   {
    float IVAR2 ;

    struct SSSS
    {
     SSSS( const float& from) {}
    };

    SSSS VAR2(IVAR2);
   }
}

// CHECK: @_ZZ25LocalTemplateFunctionTestdEN5Local3fooIdEET_S1_
int LocalTemplateFunctionTest(double d) {
  struct Local {
    template<class T> T foo(T t) {
      return t;
    }
  };
  return Local().foo(d);
}
