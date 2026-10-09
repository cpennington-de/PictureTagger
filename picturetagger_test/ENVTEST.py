import subprocess

model = subprocess.check_output(
            ["sysctl", "-n", "hw.model"],
            text=True
        ).strip()

print(model)