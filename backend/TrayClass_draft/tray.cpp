#include "tray.h"
#include <iostream>

//General Constructor
Tray::Tray(){
    TrayNum = 0;
    ChickCount = 0;
    Date = "00/00/0000";
    Time = "00:00";
}

//Constructor that takes in a tray number, date, and time
Tray::Tray(unsigned long int tNum, string d, string t){
    TrayNum = tNum;
    ChickCount = 0;
    Date = d;
    Time = t;
}

//Default destructor
Tray::~Tray(){}

//Gets Tray number
unsigned long int Tray::getTrayNum(){return TrayNum;}

//Gets ChickCount
unsigned int Tray::getChickCount(){return ChickCount;}

//Gets Date
string Tray::getDate(){return Date;}

//Gets Time
string Tray::getTime(){return Time;}

//Sets TrayNum
void Tray::setTrayNum(unsigned long int num){TrayNum = num;}

//Sets Chick Count
void Tray::setChickCount(unsigned int num){ChickCount = num;}

//Set date
void Tray::setDate(string d){Date = d;}

//Sets time
void Tray::setTime(string t){Time = t;}

//Overloads the output streaming operator
ostream& operator<<(ostream& os, const Tray& t){
    os << "Tray Number: " << t.TrayNum << endl
    << "Number of Chicks: " << t.ChickCount << endl
    << "Date of Collection: " << t.Date << endl
    << "Time of Collection: " << t.Time << endl;

    return os;
}

//Overloads the input streaming operator
istream& operator>>(istream& is, const Tray& t){
    cout << "Enter Tray Number: ";
    is >> t.TrayNum;
    cout << "Enter Number of chicks: ";
    is >> t.ChickCount;
    cout << "Enter date (MM/DD/YYYY): ";
    is >> t.Date;
    cout << "Enter time (HH:MM:SS): ";
    is >> t.Time;

    return is;
}
