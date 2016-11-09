
def enable_cv3(opencv_lib_path = "/usr/local/opencv3/lib"):
    import os, sys, pkgutil

    if "LD_LIBRARY_PATH" not in os.environ or opencv_lib_path not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] += ":" + opencv_lib_path
        try:
            if hasattr(sys.modules["__main__"], "__loader__"):
                os.execv(sys.executable, [sys.executable] + ["-m", sys.modules['__main__'].__loader__.fullname] + sys.argv)
            else:
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception, exc:
            print "Failed to load OpenCV 3:", exc
            return

    opencv_python_path = opencv_lib_path + "/python2.7/dist-packages"
    if opencv_python_path not in sys.path:
        sys.path.insert(0, opencv_python_path)

    try:
        del sys.modules['cv2']
    except AttributeError:
        pass
