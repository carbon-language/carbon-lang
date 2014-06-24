//===-- MIUtilVariant.cpp----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIUtilVariant.cpp
//
// Overview:	CMIUtilVariant implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Gotchas:		See CMIUtilVariant class description.
//
// Copyright:	None.
//--

// In-house headers:
#include "MIUtilVariant.h"  

//++ ------------------------------------------------------------------------------------
// Details:	CDataObjectBase constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMIUtilVariant::CDataObjectBase::CDataObjectBase( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CDataObjectBase copy constructor.
// Type:	Method.
// Args:	vrOther	- (R) The other object.
// Return:	None.
// Throws:	None.
//--
CMIUtilVariant::CDataObjectBase::CDataObjectBase( const CDataObjectBase & vrOther )
{
	MIunused( vrOther );
}

//++ ------------------------------------------------------------------------------------
// Details:	CDataObjectBase destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMIUtilVariant::CDataObjectBase::~CDataObjectBase( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CDataObjectBase copy operator.
// Type:	Method.
// Args:	vrOther	- (R) The other object.
// Return:	None.
// Throws:	None.
//--
CMIUtilVariant::CDataObjectBase & CMIUtilVariant::CDataObjectBase::operator= ( const CDataObjectBase & vrOther )
{
	MIunused( vrOther );
	return *this;
}

//++ ------------------------------------------------------------------------------------
// Details:	Create a new copy of *this class.
// Type:	Overrideable.
// Args:	None.
// Return:	CDataObjectBase *	- Pointer to a new object.
// Throws:	None.
//--
CMIUtilVariant::CDataObjectBase * CMIUtilVariant::CDataObjectBase::CreateCopyOfSelf( void )
{
	// Override to implement copying of variant's data object
	return new CDataObjectBase();
}

//++ ------------------------------------------------------------------------------------
// Details:	Determine if *this object is a derived from CDataObjectBase.
// Type:	Overrideable.
// Args:	None.
// Return:	bool	- True = *this is derived from CDataObjectBase, false = *this is instance of the this base class.
// Throws:	None.
//--
bool CMIUtilVariant::CDataObjectBase::GetIsDerivedClass( void ) const
{
	// Override to in the derived class and return true
	return false;
}

//++ ------------------------------------------------------------------------------------
// Details:	Perform a bitwise copy of *this object.
// Type:	Overrideable.
// Args:	vrOther	- (R) The other object.
// Return:	None.
// Throws:	None.
//--
void CMIUtilVariant::CDataObjectBase::Copy( const CDataObjectBase & vrOther )
{
	// Override to implement
	MIunused( vrOther );
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CDataObject copy constructor.
// Type:	Method.
// Args:	T		- The object's type.
//			vrOther	- (R) The other object.
// Return:	None.
// Throws:	None.
//--
template< typename T >
CMIUtilVariant::CDataObject< T >::CDataObject( const CDataObject & vrOther )
{
	if( this == &vrOther )
		return;
	CDataObjectBase::Copy( vrOther );
	Copy( vrOther );	
}

//++ ------------------------------------------------------------------------------------
// Details:	CDataObject copy operator.
// Type:	Method.
// Args:	T		- The object's type.
//			vrOther	- (R) The other object.
// Return:	None.
// Throws:	None.
//--
template< typename T >
CMIUtilVariant::CDataObject< T > & CMIUtilVariant::CDataObject< T >::operator= ( const CDataObject & vrOther )
{
	if( this == &vrOther )
		return *this;
	CDataObjectBase::Copy( vrOther );
	Copy( vrOther );
	return *this;
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMIUtilVariant constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMIUtilVariant::CMIUtilVariant( void )
:	m_pDataObject( nullptr )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	CMIUtilVariant copy constructor.
// Type:	Method.
// Args:	vrOther	- (R) The other object.
// Return:	None.
// Throws:	None.
//--
CMIUtilVariant::CMIUtilVariant( const CMIUtilVariant & vrOther )
:	m_pDataObject( nullptr )
{
	if( this == &vrOther )
		return;

	Copy( vrOther );
}

//++ ------------------------------------------------------------------------------------
// Details:	CMIUtilVariant destructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMIUtilVariant::~CMIUtilVariant( void )
{
	Destroy();
}

//++ ------------------------------------------------------------------------------------
// Details:	CMIUtilVariant copy operator.
// Type:	Method.
// Args:	vrOther	- (R) The other object.
// Return:	None.
// Throws:	None.
//--
CMIUtilVariant & CMIUtilVariant::operator= ( const CMIUtilVariant & vrOther )
{
	if( this == &vrOther )
		return *this;

	Copy( vrOther );
	return *this;
}

//++ ------------------------------------------------------------------------------------
// Details:	Release the resources used by *this object.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
void CMIUtilVariant::Destroy( void )
{
	if( m_pDataObject != nullptr )
		delete m_pDataObject;
	m_pDataObject = nullptr;
}

//++ ------------------------------------------------------------------------------------
// Details:	Bitwise copy another data object to *this variant object.
//			Because *this Variant class does not store the type of the object
//			held it performs a bitwise copy of the data object whenever the variant
//			object is copied. This could lead to issues.
// Type:	Method.
// Args:	vrOther	- (R) The other object.
// Return:	None.
// Throws:	None.
//--
void CMIUtilVariant::Copy( const CMIUtilVariant & vrOther )
{
	Destroy();
	
	if( vrOther.m_pDataObject != nullptr )
	{
		m_pDataObject = vrOther.m_pDataObject->CreateCopyOfSelf();
	}
}
