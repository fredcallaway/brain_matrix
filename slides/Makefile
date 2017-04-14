deploy:
	echo "SECRET = null" > js/multiplex.js
	rsync -e 'ssh' -a . fredc@simcoe.lmi.net:~/www.fredcallaway.com/docs/slides/distance
	echo "SECRET = '14920599571542024725'" > js/multiplex.js

figs: FORCE
	rsync -a theano:~/brain_matrix/figs/ figs

FORCE: ;