//------------------------------------
//  VisualPng.C -- Shows a PNG image
//------------------------------------

// Copyright 2000, Willem van Schaik.  For conditions of distribution and
// use, see the copyright/license/disclaimer notice in png.h

// switches

// defines

#define PROGNAME  "VisualPng"
#define LONGNAME  "Win32 Viewer for PNG-files"
#define VERSION   "1.0 of 2000 June 07"

// constants

#define MARGIN 8

// standard includes

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

// application includes

#include "png.h"
#include "pngfile.h"
#include "resource.h"

// macros

// function prototypes

LRESULT CALLBACK WndProc (HWND, UINT, WPARAM, LPARAM);
BOOL    CALLBACK AboutDlgProc (HWND, UINT, WPARAM, LPARAM) ;

BOOL CenterAbout (HWND hwndChild, HWND hwndParent);

BOOL BuildPngList (PTSTR pstrPathName, TCHAR **ppFileList, int *pFileCount,
        int *pFileIndex);

BOOL SearchPngList (TCHAR *pFileList, int FileCount, int *pFileIndex,
        PTSTR pstrPrevName, PTSTR pstrNextName);

BOOL LoadImageFile(HWND hwnd, PTSTR pstrPathName,
        png_byte **ppbImage, int *pxImgSize, int *pyImgSize, int *piChannels,
        png_color *pBkgColor);

BOOL DisplayImage (HWND hwnd, BYTE **ppDib,
        BYTE **ppDiData, int cxWinSize, int cyWinSize,
        BYTE *pbImage, int cxImgSize, int cyImgSize, int cImgChannels,
        BOOL bStretched);

BOOL InitBitmap (
        BYTE *pDiData, int cxWinSize, int cyWinSize);

BOOL FillBitmap (
        BYTE *pDiData, int cxWinSize, int cyWinSize,
        BYTE *pbImage, int cxImgSize, int cyImgSize, int cImgChannels,
        BOOL bStretched);

// a few global variables

static char *szProgName = PROGNAME;
static char *szAppName = LONGNAME;
static char *szIconName = PROGNAME;
static char szCmdFileName [MAX_PATH];

// MAIN routine

int WINAPI WinMain (HINSTANCE hInstance, HINSTANCE hPrevInstance,
                    PSTR szCmdLine, int iCmdShow)
{
    HACCEL   hAccel;
    HWND     hwnd;
    MSG      msg;
    WNDCLASS wndclass;
    int ixBorders, iyBorders;

    wndclass.style         = CS_HREDRAW | CS_VREDRAW;
    wndclass.lpfnWndProc   = WndProc;
    wndclass.cbClsExtra    = 0;
    wndclass.cbWndExtra    = 0;
    wndclass.hInstance     = hInstance;
    wndclass.hIcon         = LoadIcon (hInstance, szIconName) ;
    wndclass.hCursor       = LoadCursor (NULL, IDC_ARROW);
    wndclass.hbrBackground = NULL; // (HBRUSH) GetStockObject (GRAY_BRUSH);
    wndclass.lpszMenuName  = szProgName;
    wndclass.lpszClassName = szProgName;

    if (!RegisterClass (&wndclass))
    {
        MessageBox (NULL, TEXT ("Error: this program requires Windows NT!"),
            szProgName, MB_ICONERROR);
        return 0;
    }

    // if filename given on commandline, store it
    if ((szCmdLine != NULL) && (*szCmdLine != '\0'))
        if (szCmdLine[0] == '"')
            strncpy (szCmdFileName, szCmdLine + 1, strlen(szCmdLine) - 2);
        else
            strcpy (szCmdFileName, szCmdLine);
    else
        strcpy (szCmdFileName, "");

    // calculate size of window-borders
    ixBorders = 2 * (GetSystemMetrics (SM_CXBORDER) +
                     GetSystemMetrics (SM_CXDLGFRAME));
    iyBorders = 2 * (GetSystemMetrics (SM_CYBORDER) +
                     GetSystemMetrics (SM_CYDLGFRAME)) +
                     GetSystemMetrics (SM_CYCAPTION) +
                     GetSystemMetrics (SM_CYMENUSIZE) +
                     1; /* WvS: don't ask me why? */

    hwnd = CreateWindow (szProgName, szAppName,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        512 + 2 * MARGIN + ixBorders, 384 + 2 * MARGIN + iyBorders,
//      CW_USEDEFAULT, CW_USEDEFAULT,
        NULL, NULL, hInstance, NULL);

    ShowWindow (hwnd, iCmdShow);
    UpdateWindow (hwnd);

    hAccel = LoadAccelerators (hInstance, szProgName);

    while (GetMessage (&msg, NULL, 0, 0))
    {
        if (!TranslateAccelerator (hwnd, hAccel, &msg))
        {
            TranslateMessage (&msg);
            DispatchMessage (&msg);
        }
    }
    return msg.wParam;
}

LRESULT CALLBACK WndProc (HWND hwnd, UINT message, WPARAM wParam,
        LPARAM lParam)
{
    static HINSTANCE          hInstance ;
    static HDC                hdc;
    static PAINTSTRUCT        ps;
    static HMENU              hMenu;

    static BITMAPFILEHEADER  *pbmfh;
    static BITMAPINFOHEADER  *pbmih;
    static BYTE              *pbImage;
    static int                cxWinSize, cyWinSize;
    static int                cxImgSize, cyImgSize;
    static int                cImgChannels;
    static png_color          bkgColor = {127, 127, 127};

    static BOOL               bStretched = TRUE;

    static BYTE              *pDib = NULL;
    static BYTE              *pDiData = NULL;

    static TCHAR              szImgPathName [MAX_PATH];
    static TCHAR              szTitleName [MAX_PATH];

    static TCHAR             *pPngFileList = NULL;
    static int                iPngFileCount;
    static int                iPngFileIndex;

    BOOL                      bOk;

    switch (message)
    {
    case WM_CREATE:
        hInstance = ((LPCREATESTRUCT) lParam)->hInstance ;
        PngFileInitialize (hwnd);

        strcpy (szImgPathName, "");

        // in case we process file given on command-line

        if (szCmdFileName[0] != '\0')
        {
            strcpy (szImgPathName, szCmdFileName);

            // read the other png-files in the directory for later
            // next/previous commands

            BuildPngList (szImgPathName, &pPngFileList, &iPngFileCount,
                          &iPngFileIndex);

            // load the image from file

            if (!LoadImageFile (hwnd, szImgPathName,
                &pbImage, &cxImgSize, &cyImgSize, &cImgChannels, &bkgColor))
                return 0;

            // invalidate the client area for later update

            InvalidateRect (hwnd, NULL, TRUE);

            // display the PNG into the DIBitmap

            DisplayImage (hwnd, &pDib, &pDiData, cxWinSize, cyWinSize,
                pbImage, cxImgSize, cyImgSize, cImgChannels, bStretched);
        }

        return 0;

    case WM_SIZE:
        cxWinSize = LOWORD (lParam);
        cyWinSize = HIWORD (lParam);

        // invalidate the client area for later update

        InvalidateRect (hwnd, NULL, TRUE);

        // display the PNG into the DIBitmap

        DisplayImage (hwnd, &pDib, &pDiData, cxWinSize, cyWinSize,
            pbImage, cxImgSize, cyImgSize, cImgChannels, bStretched);

        return 0;

    case WM_INITMENUPOPUP:
        hMenu = GetMenu (hwnd);

        if (pbImage)
            EnableMenuItem (hMenu, IDM_FILE_SAVE, MF_ENABLED);
        else
            EnableMenuItem (hMenu, IDM_FILE_SAVE, MF_GRAYED);

        return 0;

    case WM_COMMAND:
        hMenu = GetMenu (hwnd);

        switch (LOWORD (wParam))
        {
        case IDM_FILE_OPEN:

            // show the File Open dialog box

            if (!PngFileOpenDlg (hwnd, szImgPathName, szTitleName))
                return 0;

            // read the other png-files in the directory for later
            // next/previous commands

            BuildPngList (szImgPathName, &pPngFileList, &iPngFileCount,
                          &iPngFileIndex);

            // load the image from file

            if (!LoadImageFile (hwnd, szImgPathName,
                &pbImage, &cxImgSize, &cyImgSize, &cImgChannels, &bkgColor))
                return 0;

            // invalidate the client area for later update

            InvalidateRect (hwnd, NULL, TRUE);

            // display the PNG into the DIBitmap

            DisplayImage (hwnd, &pDib, &pDiData, cxWinSize, cyWinSize,
                pbImage, cxImgSize, cyImgSize, cImgChannels, bStretched);

            return 0;

        case IDM_FILE_SAVE:

            // show the File Save dialog box

            if (!PngFileSaveDlg (hwnd, szImgPathName, szTitleName))
                return 0;

            // save the PNG to a disk file

            SetCursor (LoadCursor (NULL, IDC_WAIT));
            ShowCursor (TRUE);

            bOk = PngSaveImage (szImgPathName, pDiData, cxWinSize, cyWinSize,
                  bkgColor);

            ShowCursor (FALSE);
            SetCursor (LoadCursor (NULL, IDC_ARROW));

            if (!bOk)
                MessageBox (hwnd, TEXT ("Error in saving the PNG image"),
                szProgName, MB_ICONEXCLAMATION | MB_OK);
            return 0;

        case IDM_FILE_NEXT:

            // read next entry in the directory

            if (SearchPngList (pPngFileList, iPngFileCount, &iPngFileIndex,
                NULL, szImgPathName))
            {
                if (strcmp (szImgPathName, "") == 0)
                    return 0;
                
                // load the image from file
                
                if (!LoadImageFile (hwnd, szImgPathName, &pbImage,
                        &cxImgSize, &cyImgSize, &cImgChannels, &bkgColor))
                    return 0;
                
                // invalidate the client area for later update
                
                InvalidateRect (hwnd, NULL, TRUE);
                
                // display the PNG into the DIBitmap
                
                DisplayImage (hwnd, &pDib, &pDiData, cxWinSize, cyWinSize,
                    pbImage, cxImgSize, cyImgSize, cImgChannels, bStretched);
            }
            
            return 0;

        case IDM_FILE_PREVIOUS:

            // read previous entry in the directory

            if (SearchPngList (pPngFileList, iPngFileCount, &iPngFileIndex,
                szImgPathName, NULL))
            {
                
                if (strcmp (szImgPathName, "") == 0)
                    return 0;
                
                // load the image from file
                
                if (!LoadImageFile (hwnd, szImgPathName, &pbImage, &cxImgSize,
                    &cyImgSize, &cImgChannels, &bkgColor))
                    return 0;
                
                // invalidate the client area for later update
                
                InvalidateRect (hwnd, NULL, TRUE);
                
                // display the PNG into the DIBitmap
                
                DisplayImage (hwnd, &pDib, &pDiData, cxWinSize, cyWinSize,
                    pbImage, cxImgSize, cyImgSize, cImgChannels, bStretched);
            }

            return 0;

        case IDM_FILE_EXIT:

            // more cleanup needed...

            // free image buffer

            if (pDib != NULL)
            {
                free (pDib);
                pDib = NULL;
            }

            // free file-list

            if (pPngFileList != NULL)
            {
                free (pPngFileList);
                pPngFileList = NULL;
            }

            // let's go ...

            exit (0);

            return 0;

        case IDM_OPTIONS_STRETCH:
            bStretched = !bStretched;
            if (bStretched)
                CheckMenuItem (hMenu, IDM_OPTIONS_STRETCH, MF_CHECKED);
            else
                CheckMenuItem (hMenu, IDM_OPTIONS_STRETCH, MF_UNCHECKED);

            // invalidate the client area for later update

            InvalidateRect (hwnd, NULL, TRUE);

            // display the PNG into the DIBitmap

            DisplayImage (hwnd, &pDib, &pDiData, cxWinSize, cyWinSize,
                pbImage, cxImgSize, cyImgSize, cImgChannels, bStretched);

            return 0;

        case IDM_HELP_ABOUT:
            DialogBox (hInstance, TEXT ("AboutBox"), hwnd, AboutDlgProc) ;
            return 0;

        } // end switch

        break;

    case WM_PAINT:
        hdc = BeginPaint (hwnd, &ps);

        if (pDib)
            SetDIBitsToDevice (hdc, 0, 0, cxWinSize, cyWinSize, 0, 0,
                0, cyWinSize, pDiData, (BITMAPINFO *) pDib, DIB_RGB_COLORS);

        EndPaint (hwnd, &ps);
        return 0;

    case WM_DESTROY:
        if (pbmfh)
        {
            free (pbmfh);
            pbmfh = NULL;
        }

        PostQuitMessage (0);
        return 0;
    }

    return DefWindowProc (hwnd, message, wParam, lParam);
}

BOOL CALLBACK AboutDlgProc (HWND hDlg, UINT message,
                            WPARAM wParam, LPARAM lParam)
{
     switch (message)
     {
     case WM_INITDIALOG :
          ShowWindow (hDlg, SW_HIDE);
          CenterAbout (hDlg, GetWindow (hDlg, GW_OWNER));
          ShowWindow (hDlg, SW_SHOW);
          return TRUE ;

     case WM_COMMAND :
          switch (LOWORD (wParam))
          {
          case IDOK :
          case IDCANCEL :
               EndDialog (hDlg, 0) ;
               return TRUE ;
          }
          break ;
     }
     return FALSE ;
}

//---------------
//  CenterAbout
//---------------

BOOL CenterAbout (HWND hwndChild, HWND hwndParent)
{
   RECT    rChild, rParent, rWorkArea;
   int     wChild, hChild, wParent, hParent;
   int     xNew, yNew;
   BOOL  bResult;

   // Get the Height and Width of the child window
   GetWindowRect (hwndChild, &rChild);
   wChild = rChild.right - rChild.left;
   hChild = rChild.bottom - rChild.top;

   // Get the Height and Width of the parent window
   GetWindowRect (hwndParent, &rParent);
   wParent = rParent.right - rParent.left;
   hParent = rParent.bottom - rParent.top;

   // Get the limits of the 'workarea'
   bResult = SystemParametersInfo(
      SPI_GETWORKAREA,  // system parameter to query or set
      sizeof(RECT),
      &rWorkArea,
      0);
   if (!bResult) {
      rWorkArea.left = rWorkArea.top = 0;
      rWorkArea.right = GetSystemMetrics(SM_CXSCREEN);
      rWorkArea.bottom = GetSystemMetrics(SM_CYSCREEN);
   }

   // Calculate new X position, then adjust for workarea
   xNew = rParent.left + ((wParent - wChild) /2);
   if (xNew < rWorkArea.left) {
      xNew = rWorkArea.left;
   } else if ((xNew+wChild) > rWorkArea.right) {
      xNew = rWorkArea.right - wChild;
   }

   // Calculate new Y position, then adjust for workarea
   yNew = rParent.top  + ((hParent - hChild) /2);
   if (yNew < rWorkArea.top) {
      yNew = rWorkArea.top;
   } else if ((yNew+hChild) > rWorkArea.bottom) {
      yNew = rWorkArea.bottom - hChild;
   }

   // Set it, and return
   return SetWindowPos (hwndChild, NULL, xNew, yNew, 0, 0, SWP_NOSIZE |
          SWP_NOZORDER);
}

//----------------
//  BuildPngList
//----------------

BOOL BuildPngList (PTSTR pstrPathName, TCHAR **ppFileList, int *pFileCount,
     int *pFileIndex)
{
    static TCHAR              szImgPathName [MAX_PATH];
    static TCHAR              szImgFileName [MAX_PATH];
    static TCHAR              szImgFindName [MAX_PATH];

    WIN32_FIND_DATA           finddata;
    HANDLE                    hFind;

    static TCHAR              szTmp [MAX_PATH];
    BOOL                      bOk;
    int                       i, ii;
    int                       j, jj;

    // free previous file-list

    if (*ppFileList != NULL)
    {
        free (*ppFileList);
        *ppFileList = NULL;
    }

    // extract foldername, filename and search-name

    strcpy (szImgPathName, pstrPathName);
    strcpy (szImgFileName, strrchr (pstrPathName, '\\') + 1);

    strcpy (szImgFindName, szImgPathName);
    *(strrchr (szImgFindName, '\\') + 1) = '\0';
    strcat (szImgFindName, "*.png");

    // first cycle: count number of files in directory for memory allocation

    *pFileCount = 0;

    hFind = FindFirstFile(szImgFindName, &finddata);
    bOk = (hFind != (HANDLE) -1);

    while (bOk)
    {
        *pFileCount += 1;
        bOk = FindNextFile(hFind, &finddata);
    }
    FindClose(hFind);

    // allocation memory for file-list

    *ppFileList = (TCHAR *) malloc (*pFileCount * MAX_PATH);

    // second cycle: read directory and store filenames in file-list

    hFind = FindFirstFile(szImgFindName, &finddata);
    bOk = (hFind != (HANDLE) -1);

    i = 0;
    ii = 0;
    while (bOk)
    {
        strcpy (*ppFileList + ii, szImgPathName);
        strcpy (strrchr(*ppFileList + ii, '\\') + 1, finddata.cFileName);

        if (strcmp(pstrPathName, *ppFileList + ii) == 0)
            *pFileIndex = i;

        ii += MAX_PATH;
        i++;

        bOk = FindNextFile(hFind, &finddata);
    }
    FindClose(hFind);

    // finally we must sort the file-list

    for (i = 0; i < *pFileCount - 1; i++)
    {
        ii = i * MAX_PATH;
        for (j = i+1; j < *pFileCount; j++)
        {
            jj = j * MAX_PATH;
            if (strcmp (*ppFileList + ii, *ppFileList + jj) > 0)
            {
                strcpy (szTmp, *ppFileList + jj);
                strcpy (*ppFileList + jj, *ppFileList + ii);
                strcpy (*ppFileList + ii, szTmp);

                // check if this was the current image that we moved

                if (*pFileIndex == i)
                    *pFileIndex = j;
                else
                    if (*pFileIndex == j)
                        *pFileIndex = i;
            }
        }
    }

    return TRUE;
}

//----------------
//  SearchPngList
//----------------

BOOL SearchPngList (
        TCHAR *pFileList, int FileCount, int *pFileIndex,
        PTSTR pstrPrevName, PTSTR pstrNextName)
{
    if (FileCount > 0)
    {
        // get previous entry
        
        if (pstrPrevName != NULL)
        {
            if (*pFileIndex > 0)
                *pFileIndex -= 1;
            else
                *pFileIndex = FileCount - 1;
            
            strcpy (pstrPrevName, pFileList + (*pFileIndex * MAX_PATH));
        }
        
        // get next entry
        
        if (pstrNextName != NULL)
        {
            if (*pFileIndex < FileCount - 1)
                *pFileIndex += 1;
            else
                *pFileIndex = 0;
            
            strcpy (pstrNextName, pFileList + (*pFileIndex * MAX_PATH));
        }
        
        return TRUE;
    }
    else
    {
        return FALSE;
    }
}

//-----------------
//  LoadImageFile
//-----------------

BOOL LoadImageFile (HWND hwnd, PTSTR pstrPathName,
                png_byte **ppbImage, int *pxImgSize, int *pyImgSize,
                int *piChannels, png_color *pBkgColor)
{
    static TCHAR szTmp [MAX_PATH];

    // if there's an existing PNG, free the memory

    if (*ppbImage)
    {
        free (*ppbImage);
        *ppbImage = NULL;
    }

    // Load the entire PNG into memory

    SetCursor (LoadCursor (NULL, IDC_WAIT));
    ShowCursor (TRUE);

    PngLoadImage (pstrPathName, ppbImage, pxImgSize, pyImgSize, piChannels,
                  pBkgColor);

    ShowCursor (FALSE);
    SetCursor (LoadCursor (NULL, IDC_ARROW));

    if (*ppbImage != NULL)
    {
        sprintf (szTmp, "VisualPng - %s", strrchr(pstrPathName, '\\') + 1);
        SetWindowText (hwnd, szTmp);
    }
    else
    {
        MessageBox (hwnd, TEXT ("Error in loading the PNG image"),
            szProgName, MB_ICONEXCLAMATION | MB_OK);
        return FALSE;
    }

    return TRUE;
}

//----------------
//  DisplayImage
//----------------

BOOL DisplayImage (HWND hwnd, BYTE **ppDib,
        BYTE **ppDiData, int cxWinSize, int cyWinSize,
        BYTE *pbImage, int cxImgSize, int cyImgSize, int cImgChannels,
        BOOL bStretched)
{
    BYTE                       *pDib = *ppDib;
    BYTE                       *pDiData = *ppDiData;
    // BITMAPFILEHEADER        *pbmfh;
    BITMAPINFOHEADER           *pbmih;
    WORD                        wDIRowBytes;
    png_color                   bkgBlack = {0, 0, 0};
    png_color                   bkgGray  = {127, 127, 127};
    png_color                   bkgWhite = {255, 255, 255};

    // allocate memory for the Device Independant bitmap

    wDIRowBytes = (WORD) ((3 * cxWinSize + 3L) >> 2) << 2;

    if (pDib)
    {
        free (pDib);
        pDib = NULL;
    }

    if (!(pDib = (BYTE *) malloc (sizeof(BITMAPINFOHEADER) +
        wDIRowBytes * cyWinSize)))
    {
        MessageBox (hwnd, TEXT ("Error in displaying the PNG image"),
            szProgName, MB_ICONEXCLAMATION | MB_OK);
        *ppDib = pDib = NULL;
        return FALSE;
    }
    *ppDib = pDib;
    memset (pDib, 0, sizeof(BITMAPINFOHEADER));

    // initialize the dib-structure

    pbmih = (BITMAPINFOHEADER *) pDib;
    pbmih->biSize = sizeof(BITMAPINFOHEADER);
    pbmih->biWidth = cxWinSize;
    pbmih->biHeight = -((long) cyWinSize);
    pbmih->biPlanes = 1;
    pbmih->biBitCount = 24;
    pbmih->biCompression = 0;
    pDiData = pDib + sizeof(BITMAPINFOHEADER);
    *ppDiData = pDiData;

    // first fill bitmap with gray and image border

    InitBitmap (pDiData, cxWinSize, cyWinSize);

    // then fill bitmap with image

    if (pbImage)
    {
        FillBitmap (
            pDiData, cxWinSize, cyWinSize,
            pbImage, cxImgSize, cyImgSize, cImgChannels,
            bStretched);
    }

    return TRUE;
}

//--------------
//  InitBitmap
//--------------

BOOL InitBitmap (BYTE *pDiData, int cxWinSize, int cyWinSize)
{
    BYTE *dst;
    int x, y, col;

    // initialize the background with gray

    dst = pDiData;
    for (y = 0; y < cyWinSize; y++)
    {
        col = 0;
        for (x = 0; x < cxWinSize; x++)
        {
            // fill with GRAY
            *dst++ = 127;
            *dst++ = 127;
            *dst++ = 127;
            col += 3;
        }
        // rows start on 4 byte boundaries
        while ((col % 4) != 0)
        {
            dst++;
            col++;
        }
    }

    return TRUE;
}

//--------------
//  FillBitmap
//--------------

BOOL FillBitmap (
        BYTE *pDiData, int cxWinSize, int cyWinSize,
        BYTE *pbImage, int cxImgSize, int cyImgSize, int cImgChannels,
        BOOL bStretched)
{
    BYTE *pStretchedImage;
    BYTE *pImg;
    BYTE *src, *dst;
    BYTE r, g, b, a;
    const int cDIChannels = 3;
    WORD wImgRowBytes;
    WORD wDIRowBytes;
    int cxNewSize, cyNewSize;
    int cxImgPos, cyImgPos;
    int xImg, yImg;
    int xWin, yWin;
    int xOld, yOld;
    int xNew, yNew;

    if (bStretched)
    {
        cxNewSize = cxWinSize - 2 * MARGIN;
        cyNewSize = cyWinSize - 2 * MARGIN;

        // stretch the image to it's window determined size

        // the following two are the same, but the first has side-effects
        // because of rounding
//      if ((cyNewSize / cxNewSize) > (cyImgSize / cxImgSize))
        if ((cyNewSize * cxImgSize) > (cyImgSize * cxNewSize))
        {
            cyNewSize = cxNewSize * cyImgSize / cxImgSize;
            cxImgPos = MARGIN;
            cyImgPos = (cyWinSize - cyNewSize) / 2;
        }
        else
        {
            cxNewSize = cyNewSize * cxImgSize / cyImgSize;
            cyImgPos = MARGIN;
            cxImgPos = (cxWinSize - cxNewSize) / 2;
        }

        pStretchedImage = malloc (cImgChannels * cxNewSize * cyNewSize);
        pImg = pStretchedImage;

        for (yNew = 0; yNew < cyNewSize; yNew++)
        {
            yOld = yNew * cyImgSize / cyNewSize;
            for (xNew = 0; xNew < cxNewSize; xNew++)
            {
                xOld = xNew * cxImgSize / cxNewSize;

                r = *(pbImage + cImgChannels * ((yOld * cxImgSize) + xOld) + 0);
                g = *(pbImage + cImgChannels * ((yOld * cxImgSize) + xOld) + 1);
                b = *(pbImage + cImgChannels * ((yOld * cxImgSize) + xOld) + 2);
                *pImg++ = r;
                *pImg++ = g;
                *pImg++ = b;
                if (cImgChannels == 4)
                {
                    a = *(pbImage + cImgChannels * ((yOld * cxImgSize) + xOld)
                        + 3);
                    *pImg++ = a;
                }
            }
        }

        // calculate row-bytes

        wImgRowBytes = cImgChannels * cxNewSize;
        wDIRowBytes = (WORD) ((cDIChannels * cxWinSize + 3L) >> 2) << 2;

        // copy image to screen

        for (yImg = 0, yWin = cyImgPos; yImg < cyNewSize; yImg++, yWin++)
        {
            if (yWin >= cyWinSize - cyImgPos)
                break;
            src = pStretchedImage + yImg * wImgRowBytes;
            dst = pDiData + yWin * wDIRowBytes + cxImgPos * cDIChannels;

            for (xImg = 0, xWin = cxImgPos; xImg < cxNewSize; xImg++, xWin++)
            {
                if (xWin >= cxWinSize - cxImgPos)
                    break;
                r = *src++;
                g = *src++;
                b = *src++;
                *dst++ = b; /* note the reverse order */
                *dst++ = g;
                *dst++ = r;
                if (cImgChannels == 4)
                {
                    a = *src++;
                }
            }
        }

        // free memory

        if (pStretchedImage != NULL)
        {
            free (pStretchedImage);
            pStretchedImage = NULL;
        }

    }

    // process the image not-stretched

    else
    {
        // calculate the central position

        cxImgPos = (cxWinSize - cxImgSize) / 2;
        cyImgPos = (cyWinSize - cyImgSize) / 2;

        // check for image larger than window

        if (cxImgPos < MARGIN)
            cxImgPos = MARGIN;
        if (cyImgPos < MARGIN)
            cyImgPos = MARGIN;

        // calculate both row-bytes

        wImgRowBytes = cImgChannels * cxImgSize;
        wDIRowBytes = (WORD) ((cDIChannels * cxWinSize + 3L) >> 2) << 2;

        // copy image to screen

        for (yImg = 0, yWin = cyImgPos; yImg < cyImgSize; yImg++, yWin++)
        {
            if (yWin >= cyWinSize - MARGIN)
                break;
            src = pbImage + yImg * wImgRowBytes;
            dst = pDiData + yWin * wDIRowBytes + cxImgPos * cDIChannels;

            for (xImg = 0, xWin = cxImgPos; xImg < cxImgSize; xImg++, xWin++)
            {
                if (xWin >= cxWinSize - MARGIN)
                    break;
                r = *src++;
                g = *src++;
                b = *src++;
                *dst++ = b; /* note the reverse order */
                *dst++ = g;
                *dst++ = r;
                if (cImgChannels == 4)
                {
                    a = *src++;
                }
            }
        }
    }

    return TRUE;
}

//-----------------
//  end of source
//-----------------
