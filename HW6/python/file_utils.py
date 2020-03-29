import copy, os, sys

############################################################# path 
# note:
#		empty path is not valid, a path of whitespace ' ' is valid

def isstring(string_test):
	if sys.version_info[0] < 3:
		return isinstance(string_test, basestring)
	else:
		return isinstance(string_test, str)

def safe_path(input_path, warning=True, debug=True):
    '''
    convert path to a valid OS format, e.g., empty string '' to '.', remove redundant '/' at the end from 'aa/' to 'aa'

    parameters:
    	input_path:		a string

    outputs:
    	safe_data:		a valid path in OS format
    '''
    if debug: assert isstring(input_path), 'path is not a string: %s' % input_path
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data

def is_path_valid(pathname):
	try:  
		if not isstring(pathname) or not pathname: return False
	except TypeError: return False
	else: return True

def is_path_creatable(pathname):
	'''
	if any previous level of parent folder exists, returns true
	'''
	if not is_path_valid(pathname): return False
	pathname = os.path.normpath(pathname)
	pathname = os.path.dirname(os.path.abspath(pathname))

	# recursively to find the previous level of parent folder existing
	while not is_path_exists(pathname):     
		pathname_new = os.path.dirname(os.path.abspath(pathname))
		if pathname_new == pathname: return False
		pathname = pathname_new
	return os.access(pathname, os.W_OK)

def is_path_exists(pathname):
	try: return is_path_valid(pathname) and os.path.exists(pathname)
	except OSError: return False

def is_path_exists_or_creatable(pathname):
	try: return is_path_exists(pathname) or is_path_creatable(pathname)
	except OSError: return False

def isfile(pathname):
	if is_path_valid(pathname):
		pathname = os.path.normpath(pathname)
		name = os.path.splitext(os.path.basename(pathname))[0]
		ext = os.path.splitext(pathname)[1]
		return len(name) > 0 and len(ext) > 0
	else: return False;

def isfolder(pathname):
	'''
	if '.' exists in the subfolder, the function still justifies it as a folder. e.g., /mnt/dome/adhoc_0.5x/abc is a folder
	if '.' exists after all slashes, the function will not justify is as a folder. e.g., /mnt/dome/adhoc_0.5x is NOT a folder
	'''
	if is_path_valid(pathname):
		pathname = os.path.normpath(pathname)
		if pathname == './': return True
		name = os.path.splitext(os.path.basename(pathname))[0]
		ext = os.path.splitext(pathname)[1]
		return len(name) > 0 and len(ext) == 0
	else: return False

def fileparts(input_path, warning=True, debug=True):
	'''
	this function return a tuple, which contains (directory, filename, extension)
	if the file has multiple extension, only last one will be displayed

	parameters:
		input_path:     a string path

	outputs:
		directory:      the parent directory
		filename:       the file name without extension
		ext:            the extension
	'''
	good_path = safe_path(input_path, debug=debug)
	if len(good_path) == 0: return ('', '', '')
	if good_path[-1] == '/':
		if len(good_path) > 1: return (good_path[:-1], '', '')	# ignore the final '/'
		else: return (good_path, '', '')	                          # ignore the final '/'
	
	directory = os.path.dirname(os.path.abspath(good_path))
	filename = os.path.splitext(os.path.basename(good_path))[0]
	ext = os.path.splitext(good_path)[1]
	return (directory, filename, ext)

def mkdir_if_missing(input_path, warning=True, debug=True):
	'''
	create a directory if not existing:
		1. if the input is a path of file, then create the parent directory of this file
		2. if the root directory does not exists for the input, then create all the root directories recursively until the parent directory of input exists

	parameters:
		input_path:     a string path
	'''	
	good_path = safe_path(input_path, warning=warning, debug=debug)
	if debug: assert is_path_exists_or_creatable(good_path), 'input path is not valid or creatable: %s' % good_path
	dirname, _, _ = fileparts(good_path)
	if not is_path_exists(dirname): mkdir_if_missing(dirname)
	if isfolder(good_path) and not is_path_exists(good_path): os.mkdir(good_path)
