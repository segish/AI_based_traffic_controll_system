#define LED_green1 4
#define LED_yellow1 3
#define LED_red1 2
#define LED_green2 7
#define LED_yellow2 6
#define LED_red2 5
#define LED_green3 10
#define LED_yellow3 9
#define LED_red3 8
#define LED_green4 13
#define LED_yellow4 12
#define LED_red4 11
int digit[10] = {0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101, 0b1111101, 0b0000111, 0b1111111, 0b1101111};
int digit1, digit2;

void setup(){
  Serial.begin(9600);
  pinMode(LED_green1, OUTPUT);
  pinMode(LED_yellow1, OUTPUT);
  pinMode(LED_red1, OUTPUT);
  pinMode(LED_green2, OUTPUT);
  pinMode(LED_yellow2, OUTPUT);
  pinMode(LED_red2, OUTPUT);
  pinMode(LED_green3, OUTPUT);
  pinMode(LED_yellow3, OUTPUT);
  pinMode(LED_red3, OUTPUT);
  pinMode(LED_green4, OUTPUT);
  pinMode(LED_yellow4, OUTPUT);
  pinMode(LED_red4, OUTPUT);
      digitalWrite(LED_red1, HIGH);
      digitalWrite(LED_red2, HIGH);
      digitalWrite(LED_red3, HIGH);
      digitalWrite(LED_red4, HIGH);

      
  for (int i = 22; i < 29; i++) {
    pinMode(i, OUTPUT);
  }
  pinMode(30, OUTPUT);
  pinMode(29, OUTPUT);

//  dis(12);
  
}
void loop(){
  
  if(Serial.available()>0){
    String msg = Serial.readString();
    int commaIndex = msg.indexOf(',');

    String lane = msg.substring(0, commaIndex);
    String countdownStr = msg.substring(commaIndex + 1);
    int countdown_start = countdownStr.toInt();
    if(lane =="lane1"){
      lane1();
      dis(countdown_start);
    }
    if(lane =="lane2"){
      lane2();
      dis(countdown_start);
    }
    if(lane =="lane3"){
      lane3();
      dis(countdown_start);
    }
    if(lane =="lane4"){
      lane4();
      dis(countdown_start);
    }
  }else{
    back();
    lane1();
    dis(20);
    lane2();
    dis(20);
    lane3();
    dis(20);
    lane4();
    dis(20);
  }
}






void lane1(){
  
      digitalWrite(LED_red1, LOW);
      digitalWrite(LED_green4, LOW);
      digitalWrite(LED_red4, LOW);
      digitalWrite(LED_yellow1, HIGH);
      digitalWrite(LED_yellow4, HIGH);
      delay(1000);
      digitalWrite(LED_yellow1, LOW);
      digitalWrite(LED_yellow4, LOW);
      digitalWrite(LED_green1, HIGH);
      digitalWrite(LED_red4, HIGH);
}
void lane2(){
  
      digitalWrite(LED_red2, LOW);
      digitalWrite(LED_green1, LOW);
      digitalWrite(LED_yellow2, HIGH);
      digitalWrite(LED_yellow1, HIGH);
      delay(1000);
      digitalWrite(LED_yellow2, LOW);
      digitalWrite(LED_yellow1, LOW);
      digitalWrite(LED_green2, HIGH);
      digitalWrite(LED_red1, HIGH);
}
void lane3(){

      digitalWrite(LED_red3, LOW);
      digitalWrite(LED_green2, LOW);
      digitalWrite(LED_yellow3, HIGH);
      digitalWrite(LED_yellow2, HIGH);
      delay(1000);
      digitalWrite(LED_yellow3, LOW);
      digitalWrite(LED_yellow2, LOW);
      digitalWrite(LED_green3, HIGH);
      digitalWrite(LED_red2, HIGH);
}
void lane4(){

  

      digitalWrite(LED_red4, LOW);
      digitalWrite(LED_green3, LOW);
      digitalWrite(LED_yellow4, HIGH);
      digitalWrite(LED_yellow3, HIGH);
      delay(1000);
      digitalWrite(LED_yellow4, LOW);
      digitalWrite(LED_yellow3, LOW);
      digitalWrite(LED_green4, HIGH);
      digitalWrite(LED_red3, HIGH);
}


void dis(int num) {
  for (int j = num; j >= 0; j--) { 
    digit2 = j / 10;
    digit1 = j % 10;
    for (int k = 0; k < 20; k++) { 
      digitalWrite(30, HIGH);
      digitalWrite(29, LOW);
  for (int i = 22; i < 29; i++) {
    digitalWrite(i, bitRead(digit[digit2], i - 22));
  }
      delay(10);
      digitalWrite(29, HIGH);
      digitalWrite(30, LOW);
  for (int i = 22; i < 29; i++) {
    digitalWrite(i, bitRead(digit[digit1], i - 22));
  }
      delay(10);
    }
  }
}

void back(){
  
      digitalWrite(LED_yellow1, LOW);
      digitalWrite(LED_yellow2, LOW);
      digitalWrite(LED_yellow3, LOW);
      digitalWrite(LED_yellow4, LOW);
      digitalWrite(LED_green1, LOW);
      digitalWrite(LED_green2, LOW);
      digitalWrite(LED_green3, LOW);
      digitalWrite(LED_green4, LOW);
      digitalWrite(LED_red1, HIGH);
      digitalWrite(LED_red2, HIGH);
      digitalWrite(LED_red3, HIGH);
      digitalWrite(LED_red4, HIGH);
}
