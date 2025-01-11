VERSION=3.10
sudo rm -rf /Library/Frameworks/Python.framework/Versions/${VERSION}/
sudo rm -rf "/Applications/Python ${VERSION}/"
cd /usr/local/bin && ls -l | grep "/Library/Frameworks/Python.framework/Versions/${VERSION}" | awk '{print $9}' | tr -d @ | sudo xargs rm