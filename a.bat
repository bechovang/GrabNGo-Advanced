@echo off
echo ================================================
echo Git Reset Hard - Windows Version
echo ================================================
echo.

REM Bước 1: Xóa thư mục .git
echo [1/6] Xóa thư mục .git cũ...
rmdir /s /q .git
if exist .git (
    echo Lỗi: Không thể xóa .git
    pause
    exit /b 1
)
echo ✓ Đã xóa .git

REM Bước 2: Khởi tạo repo mới
echo.
echo [2/6] Khởi tạo Git repo mới...
git init
echo ✓ Đã khởi tạo repo mới

REM Bước 3: Add tất cả files
echo.
echo [3/6] Add tất cả files...
git add .
echo ✓ Đã add files

REM Bước 4: Commit
echo.
echo [4/6] Tạo commit đầu tiên...
git commit -m "Initial commit - Fresh start"
echo ✓ Đã commit

REM Bước 5: Đổi tên branch thành main
echo.
echo [5/6] Đổi tên branch thành main...
git branch -M main
echo ✓ Đã đổi tên branch

REM Bước 6: Set remote (nếu chưa có)
echo.
echo [6/6] Cấu hình remote...
git remote remove origin 2>nul
git remote add origin https://github.com/bechovang/GrabNGo-Advanced
echo ✓ Đã cấu hình remote

REM Bước 7: Force push
echo.
echo ================================================
echo Sẵn sàng force push!
echo ================================================
echo.
set /p confirm="Bạn có chắc muốn XÓA HẾT lịch sử trên GitHub? (y/n): "
if /i "%confirm%"=="y" (
    echo.
    echo Đang push...
    git push -u origin main --force
    echo.
    echo ================================================
    echo ✓ HOÀN TẤT! Đã xóa hết lịch sử cũ!
    echo ================================================
) else (
    echo.
    echo Đã hủy. Không push.
)

echo.
pause