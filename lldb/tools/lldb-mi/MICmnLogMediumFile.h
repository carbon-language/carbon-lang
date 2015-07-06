//===-- MICmnLogMediumFile.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// In-house headers:
#include "MIUtilFileStd.h"
#include "MIUtilString.h"
#include "MICmnBase.h"
#include "MICmnLog.h"
#include "MIUtilDateTimeStd.h"

//++ ============================================================================
// Details: MI common code implementation class. Logs application fn trace/message/
//          error messages to a file. Used as part of the CMICmnLog Logger
//          system. When instantiated *this object is register with the Logger
//          which the Logger when given data to write to registered medium comes
//          *this medium.
//          Singleton class.
// Gotchas: None.
// Authors: Illya Rudkin 30/01/2014.
// Changes: None.
//--
class CMICmnLogMediumFile : public CMICmnBase, public CMICmnLog::IMedium
{
    // Statics:
  public:
    static CMICmnLogMediumFile &Instance(void);

    // Methods:
  public:
    bool SetHeaderTxt(const CMIUtilString &vText);
    bool SetVerbosity(const MIuint veType);
    MIuint GetVerbosity(void) const;
    const CMIUtilString &GetFileName(void) const;
    const CMIUtilString &GetFileNamePath(void) const;
    bool IsOk(void) const;
    bool IsFileExist(void) const;
    const CMIUtilString &GetLineReturn(void) const;
    bool SetDirectory(const CMIUtilString &vPath);

    // Overridden:
  public:
    // From CMICmnBase
    /* dtor */ ~CMICmnLogMediumFile(void) override;
    // From CMICmnLog::IMedium
    bool Initialize(void) override;
    const CMIUtilString &GetName(void) const override;
    bool Write(const CMIUtilString &vData, const CMICmnLog::ELogVerbosity veType) override;
    const CMIUtilString &GetError(void) const override;
    bool Shutdown(void) override;

    // Methods:
  private:
    /* ctor */ CMICmnLogMediumFile(void);
    /* ctor */ CMICmnLogMediumFile(const CMICmnLogMediumFile &);
    void operator=(const CMICmnLogMediumFile &);

    bool FileWriteEnglish(const CMIUtilString &vData);
    bool FileFormFileNamePath(void);
    CMIUtilString MassagedData(const CMIUtilString &vData, const CMICmnLog::ELogVerbosity veType);
    bool FileWriteHeader(void);
    char ConvertLogVerbosityTypeToId(const CMICmnLog::ELogVerbosity veType) const;
    CMIUtilString ConvertCr(const CMIUtilString &vData) const;

    // Attributes:
  private:
    const CMIUtilString m_constThisMediumName;
    const CMIUtilString m_constMediumFileNameFormat;
    //
    CMIUtilString m_strMediumFileName;
    CMIUtilString m_strMediumFileDirectory;
    CMIUtilString m_fileNamePath;
    MIuint m_eVerbosityType;
    CMIUtilString m_strDate;
    CMIUtilString m_fileHeaderTxt;
    CMIUtilFileStd m_file;
    CMIUtilDateTimeStd m_dateTime;
};
