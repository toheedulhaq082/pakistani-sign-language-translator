
class User {
  final String email;
  final int userId;

  User({
    required this.email,
    required this.userId,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      email: json['email'],
      userId: json['user_id'],
    );
  }
}