//***************************************************************************
// class Unique:
//	Mixin class for classes that should never be copied.
// 
// Purpose:
//	This mixin disables both the copy constructor and the
//	assignment operator.  It also provides a default equality operator.
// 
// History:
//     09/24/96 - vadve - Created (adapted from dHPF).
//
//***************************************************************************

#ifndef UNIQUE_H
#define UNIQUE_H

#include <assert.h>


class Unique
{
protected:
  /*ctor*/	Unique		() {}
  /*dtor*/ virtual ~Unique	() {}
  
public:
  virtual bool	operator==	(const Unique& u1) const;
  virtual bool	operator!=	(const Unique& u1) const;

private:
  // 
  // Disable the copy constructor and the assignment operator
  // by making them both private:
  // 
  /*ctor*/	Unique		(Unique&)		{ assert(0); }
  virtual Unique& operator=	(const Unique& u1)	{ assert(0);
							    return *this; }
};


// Unique object equality.
inline bool
Unique::operator==(const Unique& u2) const
{
    return (bool) (this == &u2);
}


// Unique object inequality.
inline bool
Unique::operator!=(const Unique& u2) const
{
    return (bool) !(this == &u2);
}

#endif
