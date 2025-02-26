#include <iostream>
#include <string>

class Tray{
private:
    unsigned long int TrayNum;
    unsigned int ChickCount;
    string Date;
    string Time;
public:
    Tray();
    Tray(unsigned long int tNum, string d, string t);
    ~Tray();
    unsigned long int getTrayNum();
    unsigned int getChickCount();
    string getDate();
    string getTime();

    void setTrayNum(unsigned long int num);
    void setChickCount(unsigned int num);
    void setDate(string d);
    void setTime(string t);

    //overload the << operator
    friend ostream& operator<<(ostream&, const Tray&);
    friend istream& operator>>(istream&, const Tray&);
};
