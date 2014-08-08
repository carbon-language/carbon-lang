//===-- MIUtilFileStd.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MIUtilFileStd.h
//
// Overview:	CMIUtilFileStd interface.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

#pragma once

// In-house headers:
#include "MIUtilString.h"  
#include "MICmnBase.h"

//++ ============================================================================
// Details:	MI common code utility class. File handling.
// Gotchas:	None.
// Authors:	Aidan Dodds 10/03/2014.
// Changes:	None.
//--
class CMIUtilFileStd : public CMICmnBase
{
// Static:
public:
	static MIchar	GetSlash( void );

// Methods:
public:
	/* ctor */  CMIUtilFileStd( void );
	//	
	bool					CreateWrite( const CMIUtilString & vFileNamePath, bool & vwrbNewCreated );
	bool					Write( const CMIUtilString & vData );
	bool					Write( const MIchar * vpData, const MIuint vCharCnt );
	void					Close( void );
	bool					IsOk( void ) const;
	bool					IsFileExist( const CMIUtilString & vFileNamePath ) const;
	const CMIUtilString &	GetLineReturn( void ) const;
	CMIUtilString 			StripOffFileName( const CMIUtilString & vDirectoryPath ) const;

// Overridden:
public:
	// From CMICmnBase
	/* dtor */ virtual ~CMIUtilFileStd( void );
	
// Attributes:
private:
	CMIUtilString	m_fileNamePath;
	FILE *			m_pFileHandle;
	CMIUtilString	m_constCharNewLine;
	bool			m_bFileError;		// True = have a file error ATM, false = all ok
};

