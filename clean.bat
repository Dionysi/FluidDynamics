del /s /f /q .\bin\*.*
for /f %%f in ('dir /ad /b .\bin\') do rd /s /q .\bin\%%f
rmdir /s /q .\bin

del /s /f /q .\.vs\*.*
for /f %%f in ('dir /ad /b .\.vs\') do rd /s /q .\.vs\%%f
rmdir /s /q .\.vs