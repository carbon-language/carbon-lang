#ifndef REG_COLOR_MAP
#define REG_COLOR_MAP

#include <hash_map>


#ifndef VALUE_SET_H

struct hashFuncValue {                  // sturcture containing the hash func
  inline size_t operator () (const Value *const val) const 
  { return (size_t) val;  }
};

#endif


typedef int RegColorType;


class RegColorMap : hash_map <const Value *, RegColorType, hashFuncValue> 
{

 public:

  inline void setRegColor(const Value *const Val, RegColorType Col) {
    (*this)[Val] = Col;
  }


  inline RegColorType getRegColor(const Value *const Val) {
    return (*this)[Val];
  }
    

};

#endif
