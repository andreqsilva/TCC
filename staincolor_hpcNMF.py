import subprocess

def get_staincolor_hpcNMF(scheme, matrix, dir, file):
    try:
        command = [dir + file] + ["-s", scheme, "-i", matrix]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=dir)
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("stderr:", e.stderr)
        print("returncode:", e.returncode)
        raise

def main():

    # gerar o .txt do matrix

    get_staincolor_hpcNMF("ED", "V-S13-92 A1-7 B.txt", "./tests/ED/", "hpcNMF.exe")

if __name__ == "__main__":
    main()