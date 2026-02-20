@echo off
REM Quick start script for CoTracker Pipeline (Windows)

echo ============================================
echo   CoTracker Pipeline - Quick Start
echo ============================================
echo.

:menu
echo Chon mode:
echo   1. Full Pipeline (Tracking + 3D + Camera)
echo   2. Tracking Only
echo   3. Webcam Real-time
echo   4. Custom
echo   5. Exit
echo.
set /p choice="Nhap lua chon (1-5): "

if "%choice%"=="1" goto full
if "%choice%"=="2" goto tracking
if "%choice%"=="3" goto webcam
if "%choice%"=="4" goto custom
if "%choice%"=="5" goto end

echo Lua chon khong hop le!
goto menu

:full
echo.
set /p video="Nhap duong dan video (VD: ../assets/apple.mp4): "
echo.
echo Chay Full Pipeline...
python ../pipeline/pipeline.py --video_path "%video%" --mode full
goto done

:tracking
echo.
set /p video="Nhap duong dan video: "
set /p grid="Nhap grid size (mac dinh 10): "
if "%grid%"=="" set grid=10
echo.
echo Chay Tracking...
python ../pipeline/pipeline.py --video_path "%video%" --mode tracking --grid_size %grid%
goto done

:webcam
echo.
echo Khoi dong webcam...
echo (Click chuot de chon diem, R=reset, Q=thoat)
python ../pipeline/pipeline.py --mode webcam
goto done

:custom
echo.
set /p video="Video path: "
set /p mode="Mode (full/tracking/depth/camera): "
set /p output="Output dir (Enter = mac dinh): "

if "%output%"=="" (
    python ../pipeline/pipeline.py --video_path "%video%" --mode %mode%
) else (
    python ../pipeline/pipeline.py --video_path "%video%" --mode %mode% --output_dir "%output%"
)
goto done

:done
echo.
echo ============================================
echo   Hoan tat!
echo ============================================
echo.
pause
goto end

:end
