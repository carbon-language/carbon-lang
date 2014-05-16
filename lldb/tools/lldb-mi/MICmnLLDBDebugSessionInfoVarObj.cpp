//===-- MICmnLLDBDebugSessionInfoVarObj.cpp ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmnLLDBDebugSessionInfoVarObj.cpp
//
// Overview:	CMICmnLLDBDebugSessionInfoVarObj implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// In-house headers:
#include "MICmnLLDBDebugSessionInfoVarObj.h"
#include "MICmnLLDBProxySBValue.h"

// Instantiations:
const MIchar * CMICmnLLDBDebugSessionInfoVarObj::ms_aVarFormatStrings[] =
{
	// CODETAG_SESSIONINFO_VARFORMAT_ENUM
	// *** Order is import here.
	"<Invalid var format>",
	"binary",
	"octal",
	"decimal",
	"hexadecimal",
	"natural"
};
const MIchar * CMICmnLLDBDebugSessionInfoVarObj::ms_aVarFormatChars[] =
{
	// CODETAG_SESSIONINFO_VARFORMAT_ENUM
	// *** Order is import here.
	"<Invalid var format>",
	"t",
	"o",
	"d",
	"x",
	"N"
};
CMICmnLLDBDebugSessionInfoVarObj::VecVarObj_t	CMICmnLLDBDebugSessionInfoVarObj::ms_vecVarObj;
MIuint											CMICmnLLDBDebugSessionInfoVarObj::ms_nVarUniqueId = 0; // Index from 0

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnLLDBDebugSessionInfoVarObj constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnLLDBDebugSessionInfoVarObj::CMICmnLLDBDebugSessionInfoVarObj( void )
:	m_eVarFormat( eVarFormat_Natural )
,	m_eVarType( eVarType_Internal )
{
	// Do not out UpdateValue() in here as not necessary
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnLLDBDebugSessionInfoVarObj constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnLLDBDebugSessionInfoVarObj::CMICmnLLDBDebugSessionInfoVarObj( const CMIUtilString & vrStrNameReal, const CMIUtilString & vrStrName, const lldb::SBValue & vrValue )
:	m_eVarFormat( eVarFormat_Natural )
,	m_eVarType( eVarType_Internal )
,	m_strName( vrStrName )
,	m_SBValue( vrValue )
,	m_strNameReal( vrStrNameReal )
{
	UpdateValue();
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnLLDBDebugSessionInfoVarObj destructor.
// Type:	Overridable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnLLDBDebugSessionInfoVarObj::~CMICmnLLDBDebugSessionInfoVarObj( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the var format enumeration for the specified string.
// Type:	Static method.
// Args:	vrStrFormat	- (R) Text description of the var format.
// Return:	varFormat_e	- Var format enumeration.
//						- No match found return eVarFormat_Invalid.
// Throws:	None.
//--
CMICmnLLDBDebugSessionInfoVarObj::varFormat_e CMICmnLLDBDebugSessionInfoVarObj::GetVarFormatForString( const CMIUtilString & vrStrFormat )
{
	// CODETAG_SESSIONINFO_VARFORMAT_ENUM
	for( MIuint i = 0; i < eVarFormat_count; i++ )
	{
		const MIchar * pVarFormatString = ms_aVarFormatStrings[ i ];
		if( vrStrFormat == pVarFormatString )
			return static_cast< varFormat_e >( i );
	}

	return eVarFormat_Invalid;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the var format enumeration for the specified character.
// Type:	Static method.
// Args:	vrcFormat	- (R) Character representing the var format.
// Return:	varFormat_e	- Var format enumeration.
//						- No match found return eVarFormat_Invalid.
// Throws:	None.
//--
CMICmnLLDBDebugSessionInfoVarObj::varFormat_e CMICmnLLDBDebugSessionInfoVarObj::GetVarFormatForChar( const MIchar & vrcFormat )
{
	if( 'r' == vrcFormat )
		return eVarFormat_Hex;

	// CODETAG_SESSIONINFO_VARFORMAT_ENUM
	for( MIuint i = 0; i < eVarFormat_count; i++ )
	{
		const MIchar * pVarFormatChar = ms_aVarFormatChars[ i ];
		if( *pVarFormatChar == vrcFormat )
			return static_cast< varFormat_e >( i );
	}

	return eVarFormat_Invalid;
}

//++ ------------------------------------------------------------------------------------
// Details:	Return the equivalent var value formatted string for the given value type.
// Type:	Static method.
// Args:	vrValue		- (R) The var value object.
//			veVarFormat	- (R) Var format enumeration.
// Returns:	CMIUtilString	- Value formatted string.
// Throws:	None.
//--
CMIUtilString CMICmnLLDBDebugSessionInfoVarObj::GetValueStringFormatted( const lldb::SBValue & vrValue, const CMICmnLLDBDebugSessionInfoVarObj::varFormat_e veVarFormat )
{
	CMIUtilString strFormattedValue;
	
	MIuint64 nValue = 0;
	if( CMICmnLLDBProxySBValue::GetValueAsUnsigned( vrValue, nValue ) == MIstatus::success )
	{
		switch( veVarFormat )
		{
		case eVarFormat_Binary:
			strFormattedValue = CMIUtilString::FormatBinary( nValue );
			break;
		case eVarFormat_Octal:
			strFormattedValue = CMIUtilString::Format( "0%llo", nValue );
			break;
		case eVarFormat_Decimal:
			strFormattedValue = CMIUtilString::Format( "%lld", nValue );
			break;
		case eVarFormat_Hex:
			strFormattedValue = CMIUtilString::Format( "0x%llx", nValue );
			break;
		case eVarFormat_Natural:
		default:
		{
			const char * pTmp = const_cast< lldb::SBValue & >( vrValue ).GetValue();
			strFormattedValue = (pTmp != nullptr) ? pTmp : "";
		}
		}
	}
	else
	{
		// Composite variable type i.e. struct
		strFormattedValue = "{...}";
	}
	
	return strFormattedValue;
}

//++ ------------------------------------------------------------------------------------
// Details:	Delete internal container contents.
// Type:	Static method.
// Args:	None.
// Returns:	None.
// Throws:	None.
//--
void CMICmnLLDBDebugSessionInfoVarObj::VarObjClear( void )
{
	ms_vecVarObj.clear();
}

//++ ------------------------------------------------------------------------------------
// Details:	Add a var object to the internal container.
// Type:	Static method.
// Args:	vrVarObj	- (R) The var value object.
// Returns:	None.
// Throws:	None.
//--
void CMICmnLLDBDebugSessionInfoVarObj::VarObjAdd( const CMICmnLLDBDebugSessionInfoVarObj & vrVarObj )
{
	VarObjDelete( vrVarObj.GetName() );	// Be sure do not have duplicates (trouble with set so vector)
	ms_vecVarObj.push_back( vrVarObj );
}
	
//++ ------------------------------------------------------------------------------------
// Details:	Delete the var object from the internal container matching the specified name.
// Type:	Static method.
// Args:	vrVarName	- (R) The var value name.
// Returns:	None.
// Throws:	None.
//--
void CMICmnLLDBDebugSessionInfoVarObj::VarObjDelete( const CMIUtilString & vrVarName )
{
	if( vrVarName.empty() || (vrVarName == "") )
		return;

	VecVarObj_t::iterator it = ms_vecVarObj.begin();
	while( it != ms_vecVarObj.end() )
	{
		const CMICmnLLDBDebugSessionInfoVarObj & rVarObj = *it;
		const CMIUtilString & rVarName = rVarObj.GetName();
		if( rVarName == vrVarName )
		{
			ms_vecVarObj.erase( it );
			return;
		}

		// Next
		++it;
	}
}

//++ ------------------------------------------------------------------------------------
// Details:	Update an existing var object in the internal container.
// Type:	Static method.
// Args:	vrVarObj	- (R) The var value object.
// Returns:	None.
// Throws:	None.
//--
void CMICmnLLDBDebugSessionInfoVarObj::VarObjUpdate( const CMICmnLLDBDebugSessionInfoVarObj & vrVarObj )
{
	VarObjAdd( vrVarObj );
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the var object matching the specified name.
// Type:	Static method.
// Args:	vrVarName	- (R) The var value name.
//			vrwVarObj	- (W) A var object.
// Returns:	bool	- True = object found, false = object not found.
// Throws:	None.
//--
bool CMICmnLLDBDebugSessionInfoVarObj::VarObjGet( const CMIUtilString & vrVarName, CMICmnLLDBDebugSessionInfoVarObj & vrwVarObj )
{
	VecVarObj_t::iterator it = ms_vecVarObj.begin();
	while( it != ms_vecVarObj.end() )
	{
		const CMICmnLLDBDebugSessionInfoVarObj & rVarObj = *it;
		const CMIUtilString & rVarName = rVarObj.GetName();
		if( rVarName == vrVarName )
		{
			vrwVarObj = rVarObj;
			return true;
		}

		// Next
		++it;
	}

	return false;
}

//++ ------------------------------------------------------------------------------------
// Details:	A count is kept of the number of var value objects created. This is count is
//			used to ID the var value object. Reset the count to 0.
// Type:	Static method.
// Args:	None.
// Returns:	None.
// Throws:	None.
//--
void CMICmnLLDBDebugSessionInfoVarObj::VarObjIdResetToZero( void )
{
	ms_nVarUniqueId = 0;
}

//++ ------------------------------------------------------------------------------------
// Details:	A count is kept of the number of var value objects created. This is count is
//			used to ID the var value object. Increment the count by 1.
// Type:	Static method.
// Args:	None.
// Returns:	None.
// Throws:	None.
//--
void CMICmnLLDBDebugSessionInfoVarObj::VarObjIdInc( void )
{
	ms_nVarUniqueId++;
}

//++ ------------------------------------------------------------------------------------
// Details:	A count is kept of the number of var value objects created. This is count is
//			used to ID the var value object. Retrieve ID.
// Type:	Static method.
// Args:	None.
// Returns:	None.
// Throws:	None.
//--
MIuint CMICmnLLDBDebugSessionInfoVarObj::VarObjIdGet( void )
{
	return ms_nVarUniqueId;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the value formatted object's name.
// Type:	Method.
// Args:	None.
// Returns:	CMIUtilString &	- Value's var%u name text.
// Throws:	None.
//--
const CMIUtilString & CMICmnLLDBDebugSessionInfoVarObj::GetName( void ) const
{
	return m_strName;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the value formatted object's variable name as given in the MI command
//			to create the var object.
// Type:	Method.
// Args:	None.
// Returns:	CMIUtilString &	- Value's real name text.
// Throws:	None.
//--
const CMIUtilString & CMICmnLLDBDebugSessionInfoVarObj::GetNameReal( void ) const
{
	return m_strNameReal;
}
	
//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the value formatted string.
// Type:	Method.
// Args:	None.
// Returns:	CMIUtilString &	- Value formatted string.
// Throws:	None.
//--
const CMIUtilString & CMICmnLLDBDebugSessionInfoVarObj::GetValueFormatted( void ) const
{
	return m_strFormattedValue;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the LLDB Value object.
// Type:	Method.
// Args:	None.
// Returns:	lldb::SBValue &	- LLDB Value object.
// Throws:	None.
//--
const lldb::SBValue & CMICmnLLDBDebugSessionInfoVarObj::GetValue( void ) const
{
	return m_SBValue;
}

//++ ------------------------------------------------------------------------------------
// Details:	Set the var format type for *this object and upate the formatting.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnLLDBDebugSessionInfoVarObj::SetVarFormat( const varFormat_e veVarFormat )
{
	if( veVarFormat >= eVarFormat_count )
		return MIstatus::failure;

	m_eVarFormat = veVarFormat;
	UpdateValue();
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Update *this var obj. Update it's value and type.
// Type:	Method.
// Args:	None.
// Returns:	None.
// Throws:	None.
//--
void CMICmnLLDBDebugSessionInfoVarObj::UpdateValue( void )
{
	m_strFormattedValue = CMICmnLLDBDebugSessionInfoVarObj::GetValueStringFormatted( m_SBValue, m_eVarFormat );

	MIuint64 nValue = 0;
	if( CMICmnLLDBProxySBValue::GetValueAsUnsigned( m_SBValue, nValue ) == MIstatus::failure )
		m_eVarType = eVarType_Composite;

	CMICmnLLDBDebugSessionInfoVarObj::VarObjUpdate( *this );
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the enumeration type of the var object.
// Type:	Method.
// Args:	None.
// Returns:	varType_e	- Enumeration value.
// Throws:	None.
//--
CMICmnLLDBDebugSessionInfoVarObj::varType_e CMICmnLLDBDebugSessionInfoVarObj::GetType( void ) const
{
	return m_eVarType;
}
	