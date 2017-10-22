all: kk.html

clean:
	rm -rf __pycache__
	rm -rf pics
	mkdir pics
	rm t3d.md t3d.html nr_sqrt.md nr_sqrt.html kk.md kk.html

t3d.html: t3d.py
	python t3d.py --symdoc
	pandoc -t html -o t3d.html --standalone --mathjax t3d.md

nr_sqrt.html: nr_sqrt.py
	python nr_sqrt.py --symdoc
	pandoc -t html -o nr_sqrt.html --standalone --mathjax nr_sqrt.md

kk.html: kk.py
	python kk.py --symdoc
	pandoc -t html -o kk.html --standalone --mathjax kk.md
