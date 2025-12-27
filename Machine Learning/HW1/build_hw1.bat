@echo off
echo === Running all HW1 Python scripts ===
python run_all.py

echo.
echo === Cleaning old LaTeX outputs ===
del hw1_solution.aux
del hw1_solution.log
del hw1_solution.out
del hw1_solution.pdf

echo.
echo === Compiling LaTeX report ===
pdflatex -interaction=nonstopmode hw1_solution.tex
pdflatex -interaction=nonstopmode hw1_solution.tex

echo.
echo === Build finished ===
pause
